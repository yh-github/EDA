import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from classifier import HybridModel


def train_model(model: HybridModel,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: torch.device):
    """Runs one full epoch of training."""
    model.train()  # Set model to training mode
    total_loss = 0

    # 'batch' is now a TrainingSample object
    for batch in train_loader:
        # --- Change from [...] to . ---
        x_text = batch.x_text.to(device)
        x_continuous = batch.x_continuous.to(device)
        x_categorical = batch.x_categorical.to(device)
        y_true = batch.y.to(device)
        # -----------------------------

        # 1. Forward pass
        logits = model(x_text, x_continuous, x_categorical)

        # 2. Calculate loss
        loss = criterion(logits, y_true)

        # 3. Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate_model(model: HybridModel,
                   test_loader: DataLoader,
                   criterion: nn.Module,
                   device: torch.device) -> dict[str, float]:
    """Evaluates the model on the test set."""
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    all_y_true = []
    all_y_pred_proba = []

    with torch.no_grad():  # Disable gradient calculation
        for batch in test_loader:
            # --- Change from [...] to . ---
            x_text = batch.x_text.to(device)
            x_continuous = batch.x_continuous.to(device)
            x_categorical = batch.x_categorical.to(device)
            y_true = batch.y.to(device)
            # -----------------------------

            logits = model(x_text, x_continuous, x_categorical)
            loss = criterion(logits, y_true)

            total_loss += loss.item()

            # Convert logits to probabilities (0-1) using sigmoid
            probabilities = torch.sigmoid(logits)

            all_y_true.extend(y_true.cpu().numpy())
            all_y_pred_proba.extend(probabilities.cpu().numpy())

    all_y_pred = (np.array(all_y_pred_proba) >= 0.5).astype(int)

    metrics = {
        'loss': total_loss / len(test_loader),
        'accuracy': accuracy_score(all_y_true, all_y_pred),
        'f1': f1_score(all_y_true, all_y_pred, zero_division=0),
        'roc_auc': roc_auc_score(all_y_true, all_y_pred_proba)
    }

    return metrics