import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast
import lightning.pytorch as pl
import optuna
import pandas as pd
import torch
from torch.nn.functional import cross_entropy
from lightning.pytorch.callbacks import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import CrossEntropy
from torch.utils.data import DataLoader

from common.config import ExperimentConfig, FieldConfig
from common.embedder import EmbeddingService
from common.feature_processor import FeatProcParams
from tft.metrics import RawMetricsCallback, AggregatedMetricsCallback, BaseMetricsCallback

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    experiment_config: ExperimentConfig
    field_config: FieldConfig
    emb_params: EmbeddingService.Params
    feat_proc_params: FeatProcParams
    pos_weight: float | None = None
    max_epochs: int = 10
    use_aggregation: bool = False
    patience: int = 3
    reduce_on_plateau_patience: int = 2  # < patience


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

        loss_val = cross_entropy(
            y_pred.view(-1, y_pred.size(-1)),
            y_actual.view(-1),
            weight=self.weight,
            reduction="none"
        )
        return loss_val.view(y_actual.shape)

class TFTRunner:
    def __init__(
            self,
            run_config: RunConfig,
            train_ds: TimeSeriesDataSet,
            train_loader: DataLoader,
            val_loader: DataLoader,
            # Pass the validation DF here!
            val_df: pd.DataFrame | None = None
    ) -> None:
        self.run_config = run_config
        self.train_ds = train_ds
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_df = val_df
        self.best_tuning_f1: float = -1.0


    def _get_metrics_callback(self) -> BaseMetricsCallback:
        """Factory method to select the correct metrics callback."""
        if self.run_config.use_aggregation and self.val_df is not None:
            return AggregatedMetricsCallback(validation_df=self.val_df)

        if self.run_config.use_aggregation and self.val_df is None:
            logger.warning("use_aggregation=True but val_df is None. Using RawMetricsCallback.")

        return RawMetricsCallback()

    def _create_model(self, params: dict[str, Any]) -> TemporalFusionTransformer:
        loss_fn = params.get("loss")
        if loss_fn is None:
            if self.run_config.pos_weight is not None:
                loss_fn = WeightedCrossEntropy(weight=[1.0, self.run_config.pos_weight])
            else:
                loss_fn = CrossEntropy()

        tft = TemporalFusionTransformer.from_dataset(
            self.train_ds,
            learning_rate=params["learning_rate"],
            hidden_size=params["hidden_size"],
            attention_head_size=params["attention_head_size"],
            dropout=params["dropout"],
            hidden_continuous_size=params.get("hidden_continuous_size", params["hidden_size"]),
            output_size=params.get("output_size", 2),
            loss=loss_fn,
            log_interval=10,
            reduce_on_plateau_patience=self.run_config.reduce_on_plateau_patience,
        )
        return cast(TemporalFusionTransformer, tft)

    @staticmethod
    def _cleanup_memory():
        gc.collect()
        torch.cuda.empty_cache()

    def objective(self, trial: optuna.Trial, custom_search_space: dict[str, Any] | None = None,
                  best_model_save_path: str | Path | None = None) -> float:
        self._cleanup_memory()

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

            early_stop = EarlyStopping(
                monitor="val_loss",
                min_delta=1e-4,
                patience=self.run_config.patience,
                verbose=False,
                mode="min"
            )

            metric_callback = self._get_metrics_callback()

            pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

            trainer = pl.Trainer(
                max_epochs=self.run_config.max_epochs,
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
                    self.save_checkpoint(tft, params, best_model_save_path)

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


    def save_checkpoint(self, model: TemporalFusionTransformer, params: dict, path: str | Path):
        """Saves model weights AND configuration in one file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": model.state_dict(),
            "hyper_parameters": params,
            "run_config": self.run_config
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

        early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=self.run_config.patience, verbose=False, mode="min")
        metric_callback = self._get_metrics_callback()

        trainer = pl.Trainer(
            max_epochs=self.run_config.max_epochs,
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
            self.save_checkpoint(tft, params, model_path)
            # Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            # torch.save(tft.state_dict(), model_path)

        self._cleanup_memory()
        return trainer, tft

    @classmethod
    def load_from_checkpoint(cls,
        checkpoint_path: str | Path,
        dataset: TimeSeriesDataSet
    ) -> TemporalFusionTransformer:
        """
        Loads a TFT model from a custom checkpoint file created by _save_checkpoint.
        Reconstructs the model structure using the provided dataset and saved hyperparameters.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        logger.info(f"Loading custom checkpoint from {path}...")
        # Load payload (state_dict + hparams)
        payload = torch.load(path, map_location=torch.device("cpu"), weights_only=False)

        hparams = payload.get("hyper_parameters", {})
        state_dict = payload.get("state_dict", {})

        run_config = payload.get("run_config")

        if not hparams or not state_dict:
            raise ValueError("Checkpoint file is missing 'hyper_parameters' or 'state_dict'.")

        # Reconstruct Model
        # We use standard CrossEntropy for evaluation/loading unless strictly needed
        tft = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=hparams.get("learning_rate", 1e-3),
            hidden_size=hparams.get("hidden_size", 128),
            attention_head_size=hparams.get("attention_head_size", 4),
            dropout=hparams.get("dropout", 0.1),
            hidden_continuous_size=hparams.get("hidden_continuous_size", hparams.get("hidden_size", 128)),
            output_size=hparams.get("output_size", 2),
            loss=CrossEntropy(),
            log_interval=10,
            reduce_on_plateau_patience=4 if run_config is None else run_config.reduce_on_plateau_patience
        )

        tft = cast(TemporalFusionTransformer, tft)

        # Load Weights
        tft.load_state_dict(state_dict)
        tft.eval()

        logger.info("Model weights loaded successfully.")
        return tft