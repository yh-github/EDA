import logging
from pathlib import Path
import lightning.pytorch as pl
import optuna
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
from exp_utils import set_global_seed
from log_utils import setup_logging
from tft_data import build_tft_dataset, prepare_tft_data
from feature_processor import FeatProcParams
from embedder import EmbeddingService

setup_logging(Path("logs/"), "tft_tuning")
logger = logging.getLogger("tft_tuner")

# CONFIG
BATCH_SIZE = 512
MAX_EPOCHS = 20
N_TRIALS = 30
STUDY_NAME = "tft_optimization_outgoing"


# --- Custom Metrics ---

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


# --- Adapters for TorchMetrics ---
# We create distinct subclasses so PyTorch Lightning logs them with unique keys
# e.g., "val_TFTF1", "val_TFTPrecision"

class TFTMetricAdapter(torchmetrics.Metric):
    """Base adapter to fix shape mismatch between TFT output and TorchMetrics."""

    def __init__(self, metric_cls, **kwargs):
        super().__init__()
        self.metric = metric_cls(**kwargs)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds: (Batch, Prediction_Len=1, Classes) -> (Batch, Classes)
        if preds.ndim == 3:
            preds = preds.squeeze(1)
        # target: (Batch, Prediction_Len=1) -> (Batch,)
        if target.ndim == 2:
            target = target.squeeze(1)
        self.metric.update(preds, target)

    def compute(self):
        return self.metric.compute()

    def reset(self):
        self.metric.reset()


# Explicit subclasses for naming
class TFTF1(TFTMetricAdapter): pass


class TFTPrecision(TFTMetricAdapter): pass


class TFTRecall(TFTMetricAdapter): pass


class TextLogCallback(Callback):
    """Logs validation metrics to console in a readable format."""

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

        msg_parts = [f"Epoch {epoch:<2}", f"Train Loss: {train_loss:.4f}", f"Val Loss: {val_loss:.4f}"]

        # Scan for our custom metrics
        for k, v in metrics.items():
            if "val_" in k and "loss" not in k:
                val = v.item() if isinstance(v, torch.Tensor) else v

                # Clean up the name for display
                name = k.replace("val_", "").replace("TFT", "")
                msg_parts.append(f"{name}: {val:.4f}")

        logger.info(" | ".join(msg_parts))


def objective(trial, train_ds, train_loader, val_loader, pos_weight):
    # --- Hyperparameters ---
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128])
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    attention_head_size = trial.suggest_categorical("attention_head_size", [2, 4])
    gradient_clip_val = trial.suggest_float("gradient_clip_val", 0.1, 1.0)

    # --- Define Metrics ---
    # We instantiate our specific subclasses here
    metrics_list = nn.ModuleList([
        TFTF1(torchmetrics.classification.MulticlassF1Score, num_classes=2, average="weighted"),
        TFTPrecision(torchmetrics.classification.MulticlassPrecision, num_classes=2, average="weighted"),
        TFTRecall(torchmetrics.classification.MulticlassRecall, num_classes=2, average="weighted")
    ])

    tft = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_size,
        output_size=2,
        loss=WeightedCrossEntropy(weight=[1.0, pos_weight]),
        logging_metrics=metrics_list,
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
        enable_checkpointing=True
    )

    trainer.fit(model=tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # --- Retrieve Best Model Metrics ---
    val_results = trainer.validate(model=tft, dataloaders=val_loader, verbose=False)[0]

    # Capture the specific metrics we care about
    for key in ["val_TFTF1", "val_TFTPrecision", "val_TFTRecall"]:
        if key in val_results:
            clean_name = key.replace("val_TFT", "")  # e.g., F1
            trial.set_user_attr(clean_name, val_results[key])
            logger.info(f"  [Trial Final] {clean_name}: {val_results[key]:.4f}")

    return val_results["val_loss"]


if __name__ == "__main__":
    logger.info("Loading data...")
    field_config = FieldConfig()
    full_df = pd.read_csv("data/rec_data2.csv").dropna(
        subset=[field_config.date, field_config.amount, field_config.text])

    # 1. Split
    exp_params = ExperimentConfig()
    set_global_seed(exp_params.random_state)
    train_df, val_df, _ = create_train_val_test_split(test_size=0.2, val_size=0.2, full_df=full_df,
                                                      random_state=exp_params.random_state)

    # 2. Embedder
    logger.info("Initializing Embedder...")
    emb_service = EmbeddingService(model_name=EmbModel.MPNET, max_length=64, batch_size=512)

    # 3. Params
    feat_params = FeatProcParams(
        use_is_positive=False,
        use_categorical_dates=True,
        use_cyclical_dates=True,
        use_continuous_amount=True,
        use_categorical_amount=False,
        k_top=0,
        n_bins=0
    )

    # 4. Prep Train
    logger.info("Preparing Train Data...")
    train_df_prepped, pca_model, processor, meta = prepare_tft_data(
        train_df, field_config, feat_params=feat_params,
        embedding_service=emb_service, fit_processor=True
    )

    # 5. Prep Val
    logger.info("Preparing Val Data...")
    val_df_prepped, _, _, _ = prepare_tft_data(
        val_df, field_config, embedding_service=emb_service,
        pca_model=pca_model, processor=processor, fit_processor=False
    )

    # 6. Weights
    train_labels = train_df_prepped[field_config.label]
    n_pos = train_labels.sum()
    n_neg = len(train_labels) - n_pos
    pos_weight = float(n_neg / max(n_pos, 1))
    logger.info(f"Class Weight: {pos_weight:.2f}")

    # 7. Datasets
    train_ds = build_tft_dataset(train_df_prepped, field_config, meta)
    val_ds = TimeSeriesDataSet.from_dataset(train_ds, val_df_prepped, predict=True, stop_randomization=True)

    train_loader = train_ds.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=4)
    val_loader = val_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=4)

    # 8. Run
    logger.info(f"Starting Study: {STUDY_NAME}")
    study = optuna.create_study(study_name=STUDY_NAME, direction="minimize")
    study.optimize(
        lambda trial: objective(trial, train_ds, train_loader, val_loader, pos_weight),
        n_trials=N_TRIALS
    )

    logger.info("=" * 50)
    logger.info("BEST TRIAL RESULTS")
    logger.info(f"Value (Loss): {study.best_value:.4f}")
    logger.info(f"Params: {study.best_params}")
    logger.info(f"Metrics: {study.best_trial.user_attrs}")
    logger.info("=" * 50)