import time
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from classifier import HybridModel

logger = logging.getLogger(__name__)


class PyTorchTrainer:
    """
    Manages the PyTorch training and evaluation loop.
    """

    def __init__(
        self,
        model: HybridModel,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        patience: int = 3
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.patience = patience

        self.best_metric = -1.0
        self.patience_counter = 0
        self.final_metrics = {}

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Runs one full epoch of training."""
        self.model.train()
        total_loss = 0

        for batch in train_loader:
            x_text = batch.x_text.to(self.device)
            x_continuous = batch.x_continuous.to(self.device)
            x_categorical = batch.x_categorical.to(self.device)
            y_true = batch.y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x_text, x_continuous, x_categorical)
            loss = self.criterion(logits, y_true)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _evaluate(self, test_loader: DataLoader) -> dict[str, float]:
        """Evaluates the model on the test set."""
        self.model.eval()
        total_loss = 0
        all_y_true = []
        all_y_pred_proba = []

        with torch.no_grad():
            for batch in test_loader:
                x_text = batch.x_text.to(self.device)
                x_continuous = batch.x_continuous.to(self.device)
                x_categorical = batch.x_categorical.to(self.device)
                y_true = batch.y.to(self.device)

                logits = self.model(x_text, x_continuous, x_categorical)
                loss = self.criterion(logits, y_true)
                total_loss += loss.item()

                probabilities = torch.sigmoid(logits)
                all_y_true.extend(y_true.cpu().numpy())
                all_y_pred_proba.extend(probabilities.cpu().numpy())

        all_y_pred = (np.array(all_y_pred_proba) >= 0.5).astype(int)

        return {
            'loss': total_loss / len(test_loader),
            'accuracy': accuracy_score(all_y_true, all_y_pred),
            'f1': f1_score(all_y_true, all_y_pred, zero_division=0),
            'roc_auc': roc_auc_score(all_y_true, all_y_pred_proba)
        }

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int
    ) -> dict[str, float]:
        """
        Runs the full training loop with early stopping.
        """
        logger.info(f"Starting PyTorch training on {self.device} for {num_epochs} epochs...")

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            train_loss = self._train_epoch(train_loader)
            metrics = self._evaluate(test_loader)

            epoch_time = time.time() - start_time

            logger.info(
                f"Epoch {epoch}/{num_epochs} [{epoch_time:.2f}s] | "
                f"Train Loss: {train_loss:.4f} | Test Loss: {metrics['loss']:.4f} | "
                f"F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}"
            )

            # Early stopping logic
            current_metric = metrics['f1']  # Or 'roc_auc', etc.
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.final_metrics = metrics
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best F1: {self.best_metric:.4f}")
                break

        logger.info("Training complete.")
        return self.final_metrics