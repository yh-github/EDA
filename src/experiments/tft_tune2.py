import logging
import warnings
from pathlib import Path

# --- 0. Suppress Annoying Warnings ---
warnings.filterwarnings("ignore", category=FutureWarning, message=".*DataFrame.std.*")
warnings.filterwarnings("ignore", message=".*behavior of DataFrame.std with axis=None is deprecated.*")

import lightning.pytorch as pl
import optuna
from optuna.samplers import TPESampler
from optuna.integration import PyTorchLightningPruningCallback
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.callbacks import EarlyStopping, Callback
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import CrossEntropy

from config import FieldConfig, EmbModel, ExperimentConfig
from data import create_train_val_test_split
from log_utils import setup_logging
from tft_data import build_tft_dataset, prepare_tft_data
from feature_processor import FeatProcParams
from embedder import EmbeddingService

setup_logging(Path("logs/"), "tft_tuning")
logger = logging.getLogger("tft_tuner")

# --- 1. A100 Speedup: Enable Tensor Cores ---
torch.set_float32_matmul_precision('medium')

# --- CONFIG ---
MAX_ENCODER_LEN = 150
BATCH_SIZE = 2048
MAX_EPOCHS = 20
N_TRIALS = 30
STUDY_NAME = "tft_optimization_outgoing"


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


class ManualMetricCallback(Callback):
    """
    Manually collects predictions and targets to compute F1, Precision, Recall
    at the end of every validation epoch. This bypasses Lightning's logging
    complexities for pre-built modules like TFT.
    """

    def __init__(self):
        super().__init__()
        self.val_preds = []
        self.val_targets = []
        self.last_f1 = 0.0
        self.last_prec = 0.0
        self.last_rec = 0.0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # TFT outputs are dictionaries. 'prediction' key holds logits/probs depending on config
        # But usually 'output' from step is loss. We need to get predictions manually if not returned.
        # However, Lightning's validation_step for TFT returns a structure we can parse.

        # Safer way: accessing the batch directly and running a quick forward pass is expensive.
        # Better: Hook into the outputs if available.
        # PyTorch Forecasting validation_step returns a dict with 'output' (prediction) and 'x' (input).

        # NOTE: 'outputs' here is whatever validation_step returns.
        # If TFT's validation_step doesn't return raw preds, we must use the model.

        x, y = batch
        # Run lightweight inference for metrics
        with torch.no_grad():
            out = pl_module(x)
            # prediction shape: [Batch, Prediction_Length, Classes]
            preds = out['prediction']

            # Collapse time dimension if needed (Prediction_Len=1)
            if preds.ndim == 3:
                preds = preds.squeeze(1)

            # Targets are in y[0] -> [Batch, Prediction_Length]
            targets = y[0]
            if targets.ndim == 2:
                targets = targets.squeeze(1)

        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(targets.detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.val_preds:
            return

        # Concatenate all batches
        all_preds = torch.cat(self.val_preds)
        all_targets = torch.cat(self.val_targets)

        # Calculate metrics using functional API (stateless)
        # We use 'weighted' to account for class imbalance if present,
        # or 'macro'/'binary' depending on your specific need.
        # Given 2 classes, 'multiclass' with num_classes=2 works well.
        f1 = torchmetrics.functional.classification.multiclass_f1_score(
            all_preds, all_targets, num_classes=2, average="weighted"
        )
        prec = torchmetrics.functional.classification.multiclass_precision(
            all_preds, all_targets, num_classes=2, average="weighted"
        )
        rec = torchmetrics.functional.classification.multiclass_recall(
            all_preds, all_targets, num_classes=2, average="weighted"
        )

        # Store for later retrieval
        self.last_f1 = f1.item()
        self.last_prec = prec.item()
        self.last_rec = rec.item()

        # Get Losses for Logging
        train_loss = trainer.callback_metrics.get("train_loss", float("inf"))
        val_loss = trainer.callback_metrics.get("val_loss", float("inf"))

        if isinstance(train_loss, torch.Tensor): train_loss = train_loss.item()
        if isinstance(val_loss, torch.Tensor): val_loss = val_loss.item()

        logger.info(
            f"Epoch {trainer.current_epoch:<2} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"F1: {self.last_f1:.4f} | Prec: {self.last_prec:.4f} | Rec: {self.last_rec:.4f}"
        )

        # Clear buffers for next epoch
        self.val_preds = []
        self.val_targets = []


def objective(trial, train_ds, train_loader, val_loader, pos_weight):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128])
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    attention_head_size = trial.suggest_categorical("attention_head_size", [2, 4])
    gradient_clip_val = trial.suggest_float("gradient_clip_val", 0.1, 1.0)

    tft = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_size,
        output_size=2,
        loss=WeightedCrossEntropy(weight=[1.0, pos_weight]),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")

    # Use our custom callback for reliable logging
    metric_callback = ManualMetricCallback()

    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        precision="bf16-mixed",
        gradient_clip_val=gradient_clip_val,
        callbacks=[early_stop, metric_callback, pruning_callback],
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=True,
        deterministic=False
    )

    trainer.fit(model=tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Use the last computed metrics from our callback
    target_f1 = metric_callback.last_f1
    prec = metric_callback.last_prec
    rec = metric_callback.last_rec

    logger.info(f"  [Trial Final] F1: {target_f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

    trial.set_user_attr("F1", target_f1)
    trial.set_user_attr("Precision", prec)
    trial.set_user_attr("Recall", rec)

    return target_f1


if __name__ == "__main__":
    logger.info("Loading data...")
    field_config = FieldConfig()
    full_df = pd.read_csv("data/rec_data2.csv").dropna(
        subset=[field_config.date, field_config.amount, field_config.text])

    exp_params = ExperimentConfig()
    pl.seed_everything(exp_params.random_state, workers=True)

    train_df, val_df, _ = create_train_val_test_split(test_size=0.2, val_size=0.2, full_df=full_df,
                                                      random_state=exp_params.random_state)

    logger.info("Initializing Embedder...")
    emb_service = EmbeddingService(model_name=EmbModel.MPNET, max_length=64, batch_size=512)

    feat_params = FeatProcParams(
        use_is_positive=False,
        use_categorical_dates=True,
        use_cyclical_dates=True,
        use_continuous_amount=True,
        use_categorical_amount=False,
        k_top=0,
        n_bins=0
    )

    logger.info("Preparing Data...")
    train_df_prepped, pca_model, processor, meta = prepare_tft_data(
        train_df, field_config, feat_params=feat_params,
        embedding_service=emb_service, fit_processor=True
    )
    val_df_prepped, _, _, _ = prepare_tft_data(
        val_df, field_config, embedding_service=emb_service,
        pca_model=pca_model, processor=processor, fit_processor=False
    )

    train_labels = train_df_prepped[field_config.label]
    n_pos = train_labels.sum()
    n_neg = len(train_labels) - n_pos
    pos_weight = float(n_neg / max(n_pos, 1))
    logger.info(f"Class Weight: {pos_weight:.2f}")

    train_ds = build_tft_dataset(train_df_prepped, field_config, meta, max_encoder_length=MAX_ENCODER_LEN)
    val_ds = TimeSeriesDataSet.from_dataset(train_ds, val_df_prepped, predict=True, stop_randomization=True)

    train_loader = train_ds.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=12, pin_memory=True)
    val_loader = val_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=12, pin_memory=True)

    logger.info(f"Starting Study: {STUDY_NAME}")

    sampler = TPESampler(seed=exp_params.random_state)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )

    study.optimize(
        lambda trial: objective(trial, train_ds, train_loader, val_loader, pos_weight),
        n_trials=N_TRIALS
    )

    logger.info("=" * 50)
    logger.info("BEST TRIAL RESULTS")
    logger.info(f"Value (F1): {study.best_value:.4f}")
    logger.info(f"Params: {study.best_params}")
    logger.info(f"Metrics: {study.best_trial.user_attrs}")
    logger.info("=" * 50)