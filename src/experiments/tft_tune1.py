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

# UPDATE THIS for your A100
BATCH_SIZE = 1024
MAX_EPOCHS = 30
N_TRIALS = 50
STUDY_NAME = "tft_optimization_v1.3"


class TFTMulticlassF1(torchmetrics.classification.MulticlassF1Score):
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # TFT outputs (Batch, Time, Classes) -> (N, 1, C)
        # TorchMetrics expects (Batch, Classes, Time) -> (N, C, 1)
        if preds.ndim == 3 and preds.shape[1] == 1 and preds.shape[2] == self.num_classes:
            preds = preds.permute(0, 2, 1)

        super().update(preds, target)


class TextLogCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking: return
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        loss = metrics.get("val_loss", 0.0)
        if isinstance(loss, torch.Tensor): loss = loss.item()

        # Check for the new class name
        f1 = metrics.get("val_TFTMulticlassF1", 0.0)
        if isinstance(f1, torch.Tensor): f1 = f1.item()

        logger.info(f"  Epoch {epoch:<2} | Val Loss: {loss:.4f} | Val F1: {f1:.4f}")


def objective(trial, train_ds, train_loader, val_loader):
    # 1. Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    attention_head_size = trial.suggest_categorical("attention_head_size", [2, 4])
    hidden_continuous_size = trial.suggest_categorical("hidden_continuous_size", [16, 32, 64])
    gradient_clip_val = trial.suggest_float("gradient_clip_val", 0.01, 1.0)

    # 2. Model
    tft = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        output_size=2,
        loss=CrossEntropy(),
        # [NEW] Add F1 Score here
        logging_metrics=nn.ModuleList([TFTMulticlassF1(num_classes=2)]),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # 3. Trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")

    # [NEW] Add our text logger
    text_logger = TextLogCallback()

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        gradient_clip_val=gradient_clip_val,

        callbacks=[early_stop_callback, text_logger],
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False
    )

    # 4. Fit
    trainer.fit(
        model=tft,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    val_loss = trainer.callback_metrics.get("val_loss")
    return val_loss.item() if val_loss else float("inf")


if __name__ == "__main__":
    logger.info("Loading data...")
    field_config = FieldConfig()
    full_df = pd.read_csv("data/rec_data2.csv").dropna(
        subset=[field_config.date, field_config.amount, field_config.text])

    # 1. Split
    train_df, val_df, _ = create_train_val_test_split(test_size=0.2, val_size=0.2, full_df=full_df, random_state=42)

    # 2. Prep (Critical Step!)
    train_df_prepped = prepare_tft_data(train_df, field_config)
    val_df_prepped = prepare_tft_data(val_df, field_config)

    # 3. Dataset
    train_ds = build_tft_dataset(train_df_prepped, field_config)
    val_ds = TimeSeriesDataSet.from_dataset(train_ds, val_df_prepped, predict=True, stop_randomization=True)

    # 4. Loaders (Increased workers/batch size for A100)
    train_loader = train_ds.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=16)
    val_loader = val_ds.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=16)

    logger.info(f"Starting Optuna Study: {STUDY_NAME}")

    # No storage argument -> Runs in Memory
    study = optuna.create_study(study_name=STUDY_NAME, direction="minimize")

    study.optimize(
        lambda trial: objective(trial, train_ds, train_loader, val_loader),
        n_trials=N_TRIALS
    )

    logger.info("TUNING COMPLETE")
    logger.info(f"Best Params: {study.best_params}")