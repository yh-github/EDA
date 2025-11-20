import logging
from pathlib import Path

import lightning.pytorch as pl
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch.callbacks import EarlyStopping, Callback
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import CrossEntropy

from config import FieldConfig
from data import create_train_val_test_split
from log_utils import setup_logging
from tft_data import build_tft_dataset, prepare_tft_data

setup_logging(Path("logs/"), "tft_tuning")
logger = logging.getLogger("tft_tuner")

# CONFIG
BATCH_SIZE = 1024
MAX_EPOCHS = 30
N_TRIALS = 50
STUDY_NAME = "tft_optimization_v1.3"

import torch
import torch.nn.functional as F
from pytorch_forecasting.metrics import CrossEntropy


class WeightedCrossEntropy(CrossEntropy):
    """
    A custom CrossEntropy metric that accepts class weights.
    """

    def __init__(self, weight: list[float] | torch.Tensor = None, **kwargs):
        # Init parent WITHOUT weight arg to avoid the ValueError
        super().__init__(**kwargs)
        self.weight = weight

    def loss(self, y_pred, y_actual):
        # y_pred: (Batch, Time, Classes) -> View as (Batch * Time, Classes)
        # y_actual: (Batch, Time) -> View as (Batch * Time)

        # Prepare weights on the correct device
        if self.weight is not None:
            if not isinstance(self.weight, torch.Tensor):
                self.weight = torch.tensor(self.weight, device=y_pred.device, dtype=torch.float)
            else:
                self.weight = self.weight.to(y_pred.device)

        # Calculate weighted Cross Entropy
        # We use reduction='none' because TFT metrics expect the loss per-timepoint
        loss_val = F.cross_entropy(
            y_pred.view(-1, y_pred.size(-1)),
            y_actual.view(-1),
            weight=self.weight,
            reduction="none"
        )

        # Reshape back to (Batch, Time) for TFT compatibility
        return loss_val.view(y_actual.shape)


# --- [FIX] Shape Adapter for F1 Score ---
class TFTMulticlassF1(torchmetrics.classification.MulticlassF1Score):
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # TFT outputs (Batch, 1, Classes), Metric needs (Batch, Classes, 1)
        if preds.ndim == 3 and preds.shape[1] == 1:
            preds = preds.permute(0, 2, 1)
        super().update(preds, target)


# --- [FIX] Better Logging Callback ---
class TextLogCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Capture train loss at the end of the training epoch
        self.train_loss = trainer.callback_metrics.get("train_loss", float("inf"))

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        # 1. Get Val Loss
        val_loss = metrics.get("val_loss", float("inf"))
        if isinstance(val_loss, torch.Tensor): val_loss = val_loss.item()

        # 2. Get Train Loss (saved from previous hook)
        train_loss = getattr(self, "train_loss", 0.0)
        if isinstance(train_loss, torch.Tensor): train_loss = train_loss.item()

        # 3. robustly find F1 Score (Key might vary)
        f1_keys = [k for k in metrics.keys() if "F1" in k and "val" in k]
        if f1_keys:
            val_f1 = metrics[f1_keys[0]]
            if isinstance(val_f1, torch.Tensor): val_f1 = val_f1.item()
        else:
            val_f1 = -1.0  # Debug signal: Metric not found

        logger.info(
            f"  Epoch {epoch:<2} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")


def objective(trial, train_ds, train_loader, val_loader, pos_weight):
    # Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    attention_head_size = trial.suggest_categorical("attention_head_size", [2, 4])
    hidden_continuous_size = trial.suggest_categorical("hidden_continuous_size", [16, 32])
    gradient_clip_val = trial.suggest_float("gradient_clip_val", 0.01, 1.0)

    # Model
    tft = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        output_size=2,
        # Class Weights for Imbalance
        loss=WeightedCrossEntropy(weight=[1.0, pos_weight]),
        # Use our Fixed Metric
        logging_metrics=nn.ModuleList([TFTMulticlassF1(num_classes=2)]),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # Callbacks
    early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
    text_logger = TextLogCallback()

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        gradient_clip_val=gradient_clip_val,
        callbacks=[early_stop, text_logger],
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False
    )

    # Fit
    trainer.fit(model=tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    val_loss = trainer.callback_metrics.get("val_loss")
    return val_loss.item() if val_loss else float("inf")


if __name__ == "__main__":
    logger.info("Loading data...")
    field_config = FieldConfig()
    full_df = pd.read_csv("data/rec_data2.csv").dropna(
        subset=[field_config.date, field_config.amount, field_config.text])

    # Split
    train_df, val_df, _ = create_train_val_test_split(test_size=0.2, val_size=0.2, full_df=full_df, random_state=42)

    # Prep
    train_df_prepped = prepare_tft_data(train_df, field_config)
    val_df_prepped = prepare_tft_data(val_df, field_config)

    # Calc Weights
    train_labels = train_df_prepped[field_config.label]
    n_pos = train_labels.sum()
    n_neg = len(train_labels) - n_pos
    pos_weight = float(n_neg / max(n_pos, 1))
    logger.info(f"Using Class Weight: {pos_weight:.2f}")

    # Dataset & Loaders
    train_ds = build_tft_dataset(train_df_prepped, field_config)
    val_ds = TimeSeriesDataSet.from_dataset(train_ds, val_df_prepped, predict=True, stop_randomization=True)

    train_loader = train_ds.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=16)
    val_loader = val_ds.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=16)

    # Run
    logger.info(f"Starting Study: {STUDY_NAME}")
    study = optuna.create_study(study_name=STUDY_NAME, direction="minimize")
    study.optimize(
        lambda trial: objective(trial, train_ds, train_loader, val_loader, pos_weight),
        n_trials=N_TRIALS
    )
    logger.info(f"Best Params: {study.best_params}")