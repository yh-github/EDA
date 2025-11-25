import logging
from typing import Any
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torchmetrics
from lightning.pytorch.callbacks import Callback
from torchmetrics.functional.classification import binary_precision_recall_curve

logger = logging.getLogger(__name__)


class BaseMetricsCallback(Callback):
    """
    Base class for collecting validation predictions and logging metrics.
    Subclasses define how predictions are processed (raw vs. aggregated) before metric calculation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.val_preds: list[torch.Tensor] = []
        self.val_targets: list[torch.Tensor] = []

        # State for optimization tracking
        self.last_f1: float = 0.0
        self.last_prec: float = 0.0
        self.last_rec: float = 0.0

    def on_validation_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Any,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0
    ) -> None:
        x, y = batch
        with torch.no_grad():
            out = pl_module(x)
            preds = out['prediction']

            # Extract probability of class 1 (Recurring)
            if preds.ndim == 3:
                probs = torch.softmax(preds, dim=-1)[:, 0, 1]
            else:
                probs = torch.sigmoid(preds)  # Fallback if binary output

            targets = y[0]
            if targets.ndim == 2:
                targets = targets.squeeze(1)

        self.val_preds.append(probs.detach().cpu())
        self.val_targets.append(targets.detach().cpu())

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.val_preds:
            return

        # Concatenate all raw predictions
        all_probs = torch.cat(self.val_preds).numpy()
        all_targets = torch.cat(self.val_targets).numpy()

        # Cleanup memory
        self.val_preds = []
        self.val_targets = []

        # Delegate processing to subclass
        self.process_and_log(all_probs, all_targets, trainer)

    def process_and_log(self, probs: np.ndarray, targets: np.ndarray, trainer: pl.Trainer):
        """To be implemented by subclasses to handle aggregation logic."""
        raise NotImplementedError

    def _compute_and_log_metrics(self, probs: torch.Tensor | np.ndarray, targets: torch.Tensor | np.ndarray,
                                 trainer: pl.Trainer, prefix: str = ""):
        """Shared logic for calculating F1/Prec/Rec and logging."""
        if isinstance(probs, np.ndarray): probs = torch.from_numpy(probs)
        if isinstance(targets, np.ndarray): targets = torch.from_numpy(targets).long()

        # Threshold at 0.5
        preds_labels = (probs > 0.5).long()

        f1 = torchmetrics.functional.classification.binary_f1_score(preds_labels, targets)
        prec = torchmetrics.functional.classification.binary_precision(preds_labels, targets)
        rec = torchmetrics.functional.classification.binary_recall(preds_labels, targets)

        self.last_f1 = f1.item()
        self.last_prec = prec.item()
        self.last_rec = rec.item()

        precisions, recalls, thresholds = binary_precision_recall_curve(probs, targets)

        # Calculate F1 for all thresholds
        # Added epsilon to avoid division by zero
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)

        # Find max
        best_idx = torch.argmax(f1_scores)
        best_f1 = f1_scores[best_idx].item()

        # Handle edge case where best_idx is the last element (which has no corresponding threshold)
        if best_idx < len(thresholds):
            best_threshold = thresholds[best_idx].item()
        else:
            best_threshold = 1.0

        # Log for Optuna/Lightning
        self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=False, logger=False)

        train_loss = trainer.callback_metrics.get("train_loss", float("inf"))
        val_loss = trainer.callback_metrics.get("val_loss", float("inf"))

        if isinstance(train_loss, torch.Tensor): train_loss = train_loss.item()
        if isinstance(val_loss, torch.Tensor): val_loss = val_loss.item()

        logger.info(
            f"Epoch {trainer.current_epoch:<2} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"F1 (0.5): {self.last_f1:.4f} | "
            f"Best F1: {best_f1:.4f} (@ {best_threshold:.4f})"
        )


class RawMetricsCallback(BaseMetricsCallback):
    """
    Computes metrics directly on the raw predictions without grouping.
    Used for 'tune1' style experiments or when no validation dataframe is available.
    """

    def process_and_log(self, probs: np.ndarray, targets: np.ndarray, trainer: pl.Trainer):
        self._compute_and_log_metrics(probs, targets, trainer, prefix="Raw")


class AggregatedMetricsCallback(BaseMetricsCallback):
    """
    Aggregates predictions by Transaction ID (using MAX probability) before computing metrics.
    Used for 'tune2' style experiments (clustered data) to avoid double counting.
    """

    def __init__(self, validation_df: pd.DataFrame, id_col: str = "id") -> None:
        super().__init__()
        self.validation_df = validation_df
        self.id_col = id_col

    def process_and_log(self, probs: np.ndarray, targets: np.ndarray, trainer: pl.Trainer):
        # 1. Map predictions back to Transaction IDs
        # Assumes validation_df matches the dataloader order exactly.
        if len(probs) != len(self.validation_df):
            logger.warning(
                f"Shape mismatch: Preds {len(probs)} vs DF {len(self.validation_df)}. "
                f"Falling back to RAW metrics."
            )
            self._compute_and_log_metrics(probs, targets, trainer, prefix="Fallback-Raw")
            return

        # 2. Create temp DF
        temp_df = pd.DataFrame({
            'id': self.validation_df[self.id_col].values,
            'prob': probs,
            'target': targets
        })

        # 3. Aggregate: Take MAX probability per ID
        aggregated = temp_df.groupby('id').agg({
            'prob': 'max',
            'target': 'first'  # Target is constant per ID
        })

        final_probs = aggregated['prob'].values
        final_targets = aggregated['target'].values

        # 4. Compute Metrics
        self._compute_and_log_metrics(final_probs, final_targets, trainer, prefix="Agg")

