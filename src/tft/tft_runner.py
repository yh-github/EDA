import logging
import lightning.pytorch as pl
import optuna
import torch
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.callbacks import EarlyStopping, Callback
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import CrossEntropy

logger = logging.getLogger(__name__)


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
    at the end of every validation epoch.
    """

    def __init__(self):
        super().__init__()
        self.val_preds = []
        self.val_targets = []
        self.last_f1 = 0.0
        self.last_prec = 0.0
        self.last_rec = 0.0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        with torch.no_grad():
            out = pl_module(x)
            preds = out['prediction']
            if preds.ndim == 3:
                preds = preds.squeeze(1)
            targets = y[0]
            if targets.ndim == 2:
                targets = targets.squeeze(1)

        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(targets.detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.val_preds:
            return

        all_preds = torch.cat(self.val_preds)
        all_targets = torch.cat(self.val_targets)

        # Calculate metrics
        f1 = torchmetrics.functional.classification.multiclass_f1_score(
            all_preds, all_targets, num_classes=2, average="weighted"
        )
        prec = torchmetrics.functional.classification.multiclass_precision(
            all_preds, all_targets, num_classes=2, average="weighted"
        )
        rec = torchmetrics.functional.classification.multiclass_recall(
            all_preds, all_targets, num_classes=2, average="weighted"
        )

        self.last_f1 = f1.item()
        self.last_prec = prec.item()
        self.last_rec = rec.item()

        train_loss = trainer.callback_metrics.get("train_loss", float("inf"))
        val_loss = trainer.callback_metrics.get("val_loss", float("inf"))

        if isinstance(train_loss, torch.Tensor): train_loss = train_loss.item()
        if isinstance(val_loss, torch.Tensor): val_loss = val_loss.item()

        logger.info(
            f"Epoch {trainer.current_epoch:<2} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"F1: {self.last_f1:.4f} | Prec: {self.last_prec:.4f} | Rec: {self.last_rec:.4f}"
        )

        self.val_preds = []
        self.val_targets = []


class TFTRunner:
    def __init__(self, train_ds, train_loader, val_loader, pos_weight=None, max_epochs=20):
        self.train_ds = train_ds
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.pos_weight = pos_weight
        self.max_epochs = max_epochs

    def _create_model(self, params):
        loss_fn = params.get("loss")
        if loss_fn is None:
            # Default to WeightedCrossEntropy if pos_weight is provided, else standard CrossEntropy
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
            reduce_on_plateau_patience=4,
        )

    def objective(self, trial, custom_search_space=None):
        # Default search space
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128]),
            "dropout": trial.suggest_float("dropout", 0.2, 0.5),
            "attention_head_size": trial.suggest_categorical("attention_head_size", [2, 4]),
            "gradient_clip_val": trial.suggest_float("gradient_clip_val", 0.1, 1.0),
        }

        # Override or extend with custom search space if provided
        if custom_search_space:
            for key, value in custom_search_space.items():
                if callable(value):
                    params[key] = value(trial)
                else:
                    params[key] = value

        tft = self._create_model(params)

        early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
        metric_callback = ManualMetricCallback()
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

        return target_f1

    def run_tuning(self, study_name, n_trials, random_state, search_space=None):
        sampler = optuna.samplers.TPESampler(seed=random_state)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )

        study.optimize(lambda trial: self.objective(trial, search_space), n_trials=n_trials)
        return study

    def train_single(self, params):
        """Runs a single training loop without Optuna overhead."""
        tft = self._create_model(params)

        early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
        metric_callback = ManualMetricCallback()

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
        return trainer, tft