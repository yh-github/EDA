import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader


class HybridModel(nn.Module):
    """
    A PyTorch classifier that is "ablation-aware."
    It can dynamically handle any combination of text, continuous,
    and categorical features.
    """

    def __init__(self,
                 # Dims for active features
                 text_embed_dim: int,
                 continuous_feat_dim: int,

                 # Config for categorical (can be empty)
                 categorical_vocab_sizes: dict[str, int],
                 embedding_dims: dict[str, int],

                 # MLP (Classifier Head) Config
                 mlp_hidden_layers: list[int] = [128, 64],
                 dropout_rate: float = 0.3):

        super().__init__()

        # --- Store which features are active ---
        # We check if the provided dimensions are greater than 0
        self.use_text = text_embed_dim > 0
        self.use_continuous = continuous_feat_dim > 0
        self.use_categorical = len(categorical_vocab_sizes) > 0

        self.categorical_feature_names = list(categorical_vocab_sizes.keys())

        total_input_dim = 0

        # --- 1. Text Features ---
        if self.use_text:
            total_input_dim += text_embed_dim

        # --- 2. Continuous Features ---
        if self.use_continuous:
            total_input_dim += continuous_feat_dim

        # --- 3. Categorical Features ---
        total_categorical_embed_dim = 0
        if self.use_categorical:
            self.embedding_layers = nn.ModuleDict()
            for name, vocab_size in categorical_vocab_sizes.items():
                embed_dim = embedding_dims.get(name, 16)  # Default dim
                self.embedding_layers[name] = nn.Embedding(vocab_size, embed_dim)
                total_categorical_embed_dim += embed_dim
            total_input_dim += total_categorical_embed_dim

        print(
            f"Model Init: use_text={self.use_text}, use_continuous={self.use_continuous}, use_categorical={self.use_categorical}")
        print(f"Total input dim to MLP: {total_input_dim}")

        if total_input_dim == 0:
            raise ValueError("No features are enabled. Model cannot be built.")

        # --- 4. Classifier Head (MLP) ---
        layer_dims = [total_input_dim] + mlp_hidden_layers

        mlp_layers = []
        for i in range(len(layer_dims) - 1):
            mlp_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
            mlp_layers.append(nn.Dropout(dropout_rate))

        mlp_layers.append(nn.Linear(layer_dims[-1], 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self,
                x_text: torch.Tensor,
                x_continuous: torch.Tensor,
                x_categorical: torch.Tensor) -> torch.Tensor:

        # Create a list to hold all active feature tensors
        active_features = []

        # --- 1. Add Text Features (if active) ---
        if self.use_text:
            active_features.append(x_text)

        # --- 2. Add Continuous Features (if active) ---
        if self.use_continuous:
            active_features.append(x_continuous)

        # --- 3. Process and Add Categorical Features (if active) ---
        # This check prevents the IndexError
        if self.use_categorical:
            embedded_cats = []
            for i, name in enumerate(self.categorical_feature_names):
                cat_ids = x_categorical[:, i]
                embedded_cats.append(self.embedding_layers[name](cat_ids))

            all_embedded_cats = torch.cat(embedded_cats, dim=1)
            active_features.append(all_embedded_cats)

        # --- 4. Combine All Active Features ---
        combined_features = torch.cat(active_features, dim=1)

        # --- 5. Pass through Classifier Head ---
        logits = self.mlp(combined_features)

        return logits.squeeze(-1)


########
########

def train_model(model: HybridModel,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: torch.device):
    """Runs one full epoch of training."""
    model.train()
    total_loss = 0

    # 'batch' is now a TrainingSample object containing batches of data
    for batch in train_loader:
        # --- THIS IS THE ONLY CHANGE ---
        # Move data to the correct device using attribute access
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


# -----------------------------------------------------------------

def evaluate_model(model: HybridModel,
                   test_loader: DataLoader,
                   criterion: nn.Module,
                   device: torch.device) -> dict[str, float]:
    """Evaluates the model on the test set."""
    model.eval()
    total_loss = 0
    all_y_true = []
    all_y_pred_proba = []

    with torch.no_grad():
        # 'batch' is also a TrainingSample object here
        for batch in test_loader:
            x_text = batch.x_text.to(device)
            x_continuous = batch.x_continuous.to(device)
            x_categorical = batch.x_categorical.to(device)
            y_true = batch.y.to(device)
            # -----------------------------

            logits = model(x_text, x_continuous, x_categorical)
            loss = criterion(logits, y_true)

            total_loss += loss.item()

            probabilities = torch.sigmoid(logits)

            all_y_true.extend(y_true.cpu().numpy())
            all_y_pred_proba.extend(probabilities.cpu().numpy())

    # --- No changes needed from here down ---
    all_y_pred = (np.array(all_y_pred_proba) >= 0.5).astype(int)

    metrics = {
        'loss': total_loss / len(test_loader),
        'accuracy': accuracy_score(all_y_true, all_y_pred),
        'f1': f1_score(all_y_true, all_y_pred, zero_division=0),
        'roc_auc': roc_auc_score(all_y_true, all_y_pred_proba)
    }

    return metrics