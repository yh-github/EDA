import logging
from pathlib import Path

import lightning.pytorch as pl
import optuna
import pandas as pd
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, Callback
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from config import FieldConfig
from data import create_train_val_test_split
from log_utils import setup_logging
from tft_data import build_tft_dataset, prepare_tft_data
from feature_processor import FeatProcParams

from config import EmbModel
from embedder import EmbeddingService
import torch
import torch.nn.functional as F
from pytorch_forecasting.metrics import CrossEntropy
import torchmetrics

setup_logging(Path("logs/"), "tft_tuning")
logger = logging.getLogger("tft_tuner")

# CONFIG
BATCH_SIZE = 512  # Reduced slightly for safety
MAX_EPOCHS = 20
N_TRIALS = 30
STUDY_NAME = "tft_optimization_outgoing"


# --- Metrics Classes (Kept from your file) ---
class WeightedCrossEntropy(CrossEntropy):
    def __init__(self, weight: list[float] | torch.Tensor = None, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight

    def loss(self, y_pred, y_actual):
        if self.weight is not None:
            if not isinstance(self.weight, torch.Tensor):
                self.weight = torch.tensor(self.weight, device=y_pred.device, dtype=torch.float)
            else:
                self.weight = self.weight.to(y_pred.device)
        loss_val = F.cross_entropy(
            y_pred.view(-1, y_pred.size(-1)),
            y_actual.view(-1),
            weight=self.weight,
            reduction="none"
        )
        return loss_val.view(y_actual.shape)


class TFTMulticlassF1(torchmetrics.classification.MulticlassF1Score):
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.ndim == 3 and preds.shape[1] == 1:
            preds = preds.permute(0, 2, 1)
        super().update(preds, target)


class TextLogCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        self.train_loss = trainer.callback_metrics.get("train_loss", float("inf"))

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking: return
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        val_loss = metrics.get("val_loss", float("inf"))
        if isinstance(val_loss, torch.Tensor): val_loss = val_loss.item()
        train_loss = getattr(self, "train_loss", 0.0)
        if isinstance(train_loss, torch.Tensor): train_loss = train_loss.item()

        # Robustly Find F1
        val_metric_keys = [k for k in metrics.keys() if "val_" in k and "loss" not in k]
        val_f1 = metrics[val_metric_keys[0]].item() if val_metric_keys else 0.0

        logger.info(
            f"  Epoch {epoch:<2} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")


def objective(trial, train_ds, train_loader, val_loader, pos_weight):
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    attention_head_size = trial.suggest_categorical("attention_head_size", [2, 4])
    gradient_clip_val = trial.suggest_float("gradient_clip_val", 0.01, 1.0)

    tft = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_size,  # Align with model size
        output_size=2,
        loss=WeightedCrossEntropy(weight=[1.0, pos_weight]),
        logging_metrics=nn.ModuleList([TFTMulticlassF1(num_classes=2)]),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

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

    trainer.fit(model=tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    val_loss = trainer.callback_metrics.get("val_loss")
    return val_loss.item() if val_loss else float("inf")


if __name__ == "__main__":
    logger.info("Loading data...")
    field_config = FieldConfig()
    full_df = pd.read_csv("data/rec_data2.csv").dropna(
        subset=[field_config.date, field_config.amount, field_config.text])

    # 1. Split Data
    train_df, val_df, _ = create_train_val_test_split(test_size=0.2, val_size=0.2, full_df=full_df, random_state=42)

    # 2. Initialize Service for Text Embeddings
    logger.info("Initializing Embedding Service...")
    emb_service = EmbeddingService(model_name=EmbModel.MPNET, max_length=64, batch_size=512)

    # 3. Define Feature Params (Rich Features)
    feat_params = FeatProcParams(
        use_cyclical_dates=True,
        use_continuous_amount=True,
        use_categorical_amount=True,  # Use tokens for TFT
        k_top=50,
        n_bins=20
    )

    # 4. Prepare TRAIN Data
    # - Sorts Data
    # - Generates Embeddings (aligned)
    # - Fits Processor
    logger.info("Preparing Train Data (Fitting Processor + PCA)...")
    train_df_prepped, pca_model, processor, meta = prepare_tft_data(
        train_df,
        field_config,
        feat_params=feat_params,
        embedding_service=emb_service,
        fit_processor=True
    )

    # 5. Prepare VAL Data
    # - Uses fitted PCA and Processor
    logger.info("Preparing Val Data...")
    val_df_prepped, _, _, _ = prepare_tft_data(
        val_df,
        field_config,
        embedding_service=emb_service,
        pca_model=pca_model,
        processor=processor,
        fit_processor=False
    )

    # 6. Calculate Class Weights (on filtered outgoing data)
    train_labels = train_df_prepped[field_config.label]
    n_pos = train_labels.sum()
    n_neg = len(train_labels) - n_pos
    pos_weight = float(n_neg / max(n_pos, 1))
    logger.info(f"Data Filtered. New Class Weight: {pos_weight:.2f}")

    # 7. Build Datasets
    # Pass metadata so it knows which columns are reals/categoricals
    train_ds = build_tft_dataset(train_df_prepped, field_config, meta)
    val_ds = TimeSeriesDataSet.from_dataset(train_ds, val_df_prepped, predict=True, stop_randomization=True)

    train_loader = train_ds.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=4)
    val_loader = val_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=4)

    # 8. Run Optuna
    logger.info(f"Starting Study: {STUDY_NAME}")
    study = optuna.create_study(study_name=STUDY_NAME, direction="minimize")
    study.optimize(
        lambda trial: objective(trial, train_ds, train_loader, val_loader, pos_weight),
        n_trials=N_TRIALS
    )
    logger.info(f"Best Params: {study.best_params}")