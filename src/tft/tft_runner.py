import logging
import gc
from pathlib import Path
from typing import Any, Callable

import lightning.pytorch as pl
import optuna
import torch
import torch.nn.functional as F
import torchmetrics
import pandas as pd
import numpy as np
from lightning.pytorch.callbacks import EarlyStopping, Callback
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import CrossEntropy
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class WeightedCrossEntropy(CrossEntropy):
    def __init__(self, weight: list[float] | torch.Tensor | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.weight: list[float] | torch.Tensor | None = weight

    def loss(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
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


class AggregatedMetricCallback(Callback):
    """
    Collects predictions on overlapping/expanded data, aggregates them by
    Transaction ID (or unique grouping key), and THEN computes metrics.

    This ensures we don't double-count transactions that appear in multiple bins.
    """

    def __init__(self, validation_df: pd.DataFrame | None = None, id_col: str = "id") -> None:
        """
        Args:
            validation_df: The DataFrame used to create the validation set.
                           Must contain the `id_col` to map predictions back to unique txns.
            id_col: The column name representing the unique transaction ID.
        """
        super().__init__()
        self.val_preds: list[torch.Tensor] = []
        self.val_targets: list[torch.Tensor] = []
        self.validation_df = validation_df
        self.id_col = id_col

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
            # Shape: [Batch, Prediction Length (1), Classes (2)]
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

        # 1. Concatenate all raw predictions (Expanded Space)
        all_probs = torch.cat(self.val_preds).numpy()
        all_targets = torch.cat(self.val_targets).numpy()

        # If we don't have the original DF, we can't aggregate, so fall back to raw metrics
        if self.validation_df is None:
            logger.warning(
                "No validation_df provided to callback. Metrics will be calculated on EXPANDED data (inaccurate).")
            self._compute_and_log(all_probs, all_targets, trainer)
            return

        # 2. Map predictions back to Transaction IDs
        # The validation loader iterates sequentially. We assume the order matches `validation_df`.
        # CRITICAL: This assumes `validation_df` was passed in the exact same sort order as the DataSet.
        if len(all_probs) != len(self.validation_df):
            # This happens if DataLoader drops the last batch or similar.
            # Ideally, use proper indexing from `x` in `on_validation_batch_end` if available.
            logger.warning(
                f"Shape mismatch: Preds {len(all_probs)} vs DF {len(self.validation_df)}. Skipping aggregation.")
            self._compute_and_log(all_probs, all_targets, trainer)
            return

        # Create a temp DF for aggregation
        temp_df = pd.DataFrame({
            'id': self.validation_df[self.id_col].values,
            'prob': all_probs,
            'target': all_targets
        })

        # 3. Resolution Strategy: MAX Probability
        # If a txn is in multiple bins, we take the highest confidence that it IS recurring.
        # Alternatively, you could take MEAN.
        aggregated = temp_df.groupby('id').agg({
            'prob': 'max',
            'target': 'first'  # Target should be same across duplicates
        })

        final_probs = torch.tensor(aggregated['prob'].values)
        final_targets = torch.tensor(aggregated['target'].values).long()

        # 4. Compute Metrics on UNIQUE Transactions
        self._compute_and_log(final_probs, final_targets, trainer)

        # Cleanup
        self.val_preds = []
        self.val_targets = []

    def _compute_and_log(self, probs: torch.Tensor | np.ndarray, targets: torch.Tensor | np.ndarray,
                         trainer: pl.Trainer):
        if isinstance(probs, np.ndarray): probs = torch.from_numpy(probs)
        if isinstance(targets, np.ndarray): targets = torch.from_numpy(targets).long()

        # Convert probs to labels for F1/Prec/Rec
        preds_labels = (probs > 0.5).long()

        f1 = torchmetrics.functional.classification.binary_f1_score(preds_labels, targets)
        prec = torchmetrics.functional.classification.binary_precision(preds_labels, targets)
        rec = torchmetrics.functional.classification.binary_recall(preds_labels, targets)

        self.last_f1 = f1.item()
        self.last_prec = prec.item()
        self.last_rec = rec.item()

        self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=False, logger=False)

        train_loss = trainer.callback_metrics.get("train_loss", float("inf"))
        val_loss = trainer.callback_metrics.get("val_loss", float("inf"))

        if isinstance(train_loss, torch.Tensor): train_loss = train_loss.item()
        if isinstance(val_loss, torch.Tensor): val_loss = val_loss.item()

        logger.info(
            f"Epoch {trainer.current_epoch:<2} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"F1 (Agg): {self.last_f1:.4f} | Prec: {self.last_prec:.4f} | Rec: {self.last_rec:.4f}"
        )


class TFTRunner:
    def __init__(
            self,
            train_ds: TimeSeriesDataSet,
            train_loader: DataLoader,
            val_loader: DataLoader,
            # Pass the validation DF here!
            val_df: pd.DataFrame | None = None,
            pos_weight: float | None = None,
            max_epochs: int = 10
    ) -> None:
        self.train_ds = train_ds
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_df = val_df
        self.pos_weight = pos_weight
        self.max_epochs = max_epochs
        self.best_tuning_f1: float = -1.0

        # TODO parameters
        self.patience = 3
        self.reduce_on_plateau_patience = 2 # < patience

    def _create_model(self, params: dict[str, Any]) -> TemporalFusionTransformer:
        loss_fn = params.get("loss")
        if loss_fn is None:
            if self.pos_weight is not None:
                loss_fn = WeightedCrossEntropy(weight=[1.0, self.pos_weight])
            else:
                loss_fn = CrossEntropy()

        return TemporalFusionTransformer.from_dataset(
            self.train_ds,
            learning_rate=params["learning_rate"],
            hidden_size=params["hidden_size"],
            attention_head_size=params["attention_head_size"],
            dropout=params["dropout"],
            hidden_continuous_size=params.get("hidden_continuous_size", params["hidden_size"]),
            output_size=params.get("output_size", 2),
            loss=loss_fn,
            log_interval=10,
            reduce_on_plateau_patience=self.reduce_on_plateau_patience,
        )

    def _cleanup_memory(self):
        gc.collect()
        torch.cuda.empty_cache()

    def objective(self, trial: optuna.Trial, custom_search_space: dict[str, Any] | None = None,
                  best_model_save_path: str | Path | None = None) -> float:
        self._cleanup_memory()

        # ... (Param suggestion code same as before) ...
        custom_search_space = custom_search_space or {}

        def suggest(name: str, default_fn: Callable[[], Any]) -> Any:
            if name in custom_search_space:
                val = custom_search_space[name]
                return val(trial) if callable(val) else val
            return default_fn()

        params: dict[str, Any] = {}
        params["learning_rate"] = suggest("learning_rate",
                                          lambda: trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True))
        params["hidden_size"] = suggest("hidden_size", lambda: trial.suggest_categorical("hidden_size", [64, 128]))
        params["dropout"] = suggest("dropout", lambda: trial.suggest_float("dropout", 0.2, 0.5))
        params["attention_head_size"] = suggest("attention_head_size",
                                                lambda: trial.suggest_categorical("attention_head_size", [2, 4]))
        params["gradient_clip_val"] = suggest("gradient_clip_val",
                                              lambda: trial.suggest_float("gradient_clip_val", 0.1, 1.0))

        try:
            tft = self._create_model(params)

            early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=self.patience, verbose=False, mode="min")

            metric_callback = AggregatedMetricCallback(validation_df=self.val_df)

            pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                accelerator="auto",
                precision="bf16-mixed",
                gradient_clip_val=params.get("gradient_clip_val", 0.1),
                callbacks=[early_stop, metric_callback, pruning_callback],
                enable_progress_bar=False,
                logger=False,
                enable_checkpointing=True,
                deterministic=False
            )

            trainer.fit(model=tft, train_dataloaders=self.train_loader, val_dataloaders=self.val_loader)

            target_f1 = metric_callback.last_f1
            prec = metric_callback.last_prec
            rec = metric_callback.last_rec

            logger.info(f"  [Trial Final] F1: {target_f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

            trial.set_user_attr("F1", target_f1)
            trial.set_user_attr("Precision", prec)
            trial.set_user_attr("Recall", rec)

            if target_f1 > self.best_tuning_f1:
                self.best_tuning_f1 = target_f1
                if best_model_save_path:
                    logger.info(f"New best F1 ({target_f1:.4f})! Saving model to {best_model_save_path}")
                    self._save_checkpoint(tft, params, best_model_save_path)

            return target_f1

        except torch.cuda.OutOfMemoryError:
            logger.warning("Trial failed due to CUDA OutOfMemoryError. Pruning trial.")
            self._cleanup_memory()
            raise optuna.TrialPruned("CUDA OutOfMemoryError")

        except Exception as e:
            logger.error(f"Trial failed with unexpected error: {e}")
            self._cleanup_memory()
            raise e

        finally:
            self._cleanup_memory()

    # ... (run_tuning and train_single updated to use _cleanup_memory are assumed present) ...
    def run_tuning(
            self,
            study_name: str,
            n_trials: int,
            random_state: int,
            search_space: dict[str, Any] | None = None,
            best_model_save_path: str | Path | None = None
    ) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(seed=random_state)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )

        study.optimize(
            lambda trial: self.objective(trial, search_space, best_model_save_path),
            n_trials=n_trials,
            catch=(torch.cuda.OutOfMemoryError,)
        )
        return study

    @staticmethod
    def _save_checkpoint(model: TemporalFusionTransformer, params: dict, path: str | Path):
        """Saves model weights AND configuration in one file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": model.state_dict(),
            "hyper_parameters": params
        }
        torch.save(payload, path)


    def train_single(
            self,
            params: dict[str, Any],
            model_path: str | Path | None = None
    ) -> tuple[pl.Trainer, TemporalFusionTransformer]:
        self._cleanup_memory()

        tft = self._create_model(params)

        if model_path:
            path_obj = Path(model_path)
            if path_obj.exists():
                logger.info(f"Found checkpoint at {path_obj}. Loading...")
                state_dict = torch.load(path_obj)
                tft.load_state_dict(state_dict)
                tft.eval()
                trainer = pl.Trainer(accelerator="auto", logger=False, enable_checkpointing=False, max_epochs=0)
                return trainer, tft

        early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=self.patience, verbose=False, mode="min")
        metric_callback = AggregatedMetricCallback(validation_df=self.val_df)

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="auto",
            precision="bf16-mixed",
            gradient_clip_val=params.get("gradient_clip_val", 0.1),
            callbacks=[early_stop, metric_callback],
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False
        )

        trainer.fit(model=tft, train_dataloaders=self.train_loader, val_dataloaders=self.val_loader)

        if model_path:
            logger.info(f"Saving model to {model_path}...")
            self._save_checkpoint(tft, params, model_path)
            # Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            # torch.save(tft.state_dict(), model_path)

        self._cleanup_memory()
        return trainer, tft