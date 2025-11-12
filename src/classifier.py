import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader


class HybridModel(nn.Module):
    """
    A PyTorch classifier that combines pre-computed text embeddings,
    continuous numerical features, and categorical features (via nn.Embedding).
    """

    def __init__(self,
                 # 1. Text Embedding Config
                 text_embed_dim: int,

                 # 2. Continuous Features Config
                 continuous_feat_dim: int,

                 # 3. Categorical Features Config (Vocab sizes and embedding dims)
                 categorical_vocab_sizes: dict[str, int],
                 embedding_dims: dict[str, int],

                 # 4. MLP (Classifier Head) Config
                 mlp_hidden_layers: list[int] = [128, 64],
                 dropout_rate: float = 0.3):

        super().__init__()

        self.categorical_feature_names = list(categorical_vocab_sizes.keys())

        # --- 1. Embedding Layers for Categorical Features ---
        self.embedding_layers = nn.ModuleDict()
        total_categorical_embed_dim = 0

        for name, vocab_size in categorical_vocab_sizes.items():
            embed_dim = embedding_dims.get(name, 16)  # Default to 16 if not specified
            self.embedding_layers[name] = nn.Embedding(vocab_size, embed_dim)
            total_categorical_embed_dim += embed_dim

        print(
            f"Created {len(self.embedding_layers)} embedding layers. Total categorical embed dim: {total_categorical_embed_dim}")

        # --- 2. Calculate Total Input Dimension for the MLP ---
        total_input_dim = text_embed_dim + continuous_feat_dim + total_categorical_embed_dim
        print(f"Total input dim to MLP: {total_input_dim}")

        # --- 3. Classifier Head (MLP) ---
        layer_dims = [total_input_dim] + mlp_hidden_layers

        mlp_layers = []
        for i in range(len(layer_dims) - 1):
            mlp_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.BatchNorm1d(layer_dims[i + 1]))  # BatchNorm is very helpful
            mlp_layers.append(nn.Dropout(dropout_rate))

        # Add the final output layer (1 logit for binary classification)
        mlp_layers.append(nn.Linear(layer_dims[-1], 1))

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self,
                x_text: torch.Tensor,
                x_continuous: torch.Tensor,
                x_categorical: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass.

        :param x_text: (batch_size, text_embed_dim) - From EmbeddingService
        :param x_continuous: (batch_size, continuous_feat_dim) - Cyclical dates, log_amount, etc.
        :param x_categorical: (batch_size, num_categorical_features) - Integer IDs
        :return: (batch_size,) - Raw logits
        """

        # --- 1. Process Categorical Features ---
        # Look up embeddings for each categorical feature
        embedded_cats = []
        for i, name in enumerate(self.categorical_feature_names):
            # Get the i-th column of categorical IDs
            cat_ids = x_categorical[:, i]
            embedded_cats.append(self.embedding_layers[name](cat_ids))

        # Concatenate all looked-up embeddings
        all_embedded_cats = torch.cat(embedded_cats, dim=1)  # (batch_size, total_categorical_embed_dim)

        # --- 2. Combine All Feature Types ---
        combined_features = torch.cat([
            x_text,
            x_continuous,
            all_embedded_cats
        ], dim=1)  # (batch_size, total_input_dim)

        # --- 3. Pass through Classifier Head ---
        logits = self.mlp(combined_features)

        # Squeeze to remove the last dimension (from [batch_size, 1] to [batch_size])
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
            # --- THIS IS THE ONLY CHANGE ---
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