from dataclasses import dataclass, field
from typing import Self

import torch
import torch.nn as nn

from data import FeatureSet
from feature_processor import FeatureMetadata, FeatureHyperParams


@dataclass(frozen=True)
class HybridModelConfig:
    """
    Holds all feature-related parameters for initializing the HybridModel.
    """
    text_embed_dim: int = 0
    continuous_feat_dim: int = 0
    categorical_vocab_sizes: dict[str, int] = field(default_factory=dict)
    embedding_dims: dict[str, int] = field(default_factory=dict)


class HybridModel(nn.Module):

    @dataclass(frozen=True)
    class MlpHyperParams:
        """Holds all hyperparameters for the MLP classifier head."""
        mlp_hidden_layers: list[int] = field(default_factory=lambda: [128, 64])
        dropout_rate: float = 0.3

    def __init__(
        self,
        config: FeatureHyperParams,
        mlp_config: MlpHyperParams
    ):

        super().__init__()

        # --- Store which features are active based on the config ---
        self.use_text = config.text_embed_dim > 0
        self.use_continuous = config.continuous_feat_dim > 0
        self.use_categorical = len(config.categorical_vocab_sizes) > 0

        self.categorical_feature_names = list(config.categorical_vocab_sizes.keys())

        total_input_dim = 0

        # --- 1. Text Features ---
        if self.use_text:
            total_input_dim += config.text_embed_dim

        # --- 2. Continuous Features ---
        if self.use_continuous:
            total_input_dim += config.continuous_feat_dim

        # --- 3. Categorical Features ---
        total_categorical_embed_dim = 0
        if self.use_categorical:
            self.embedding_layers = nn.ModuleDict()
            for name, vocab_size in config.categorical_vocab_sizes.items():
                embed_dim = config.embedding_dims.get(name, 16)  # Default dim
                self.embedding_layers[name] = nn.Embedding(vocab_size, embed_dim)
                total_categorical_embed_dim += embed_dim
            total_input_dim += total_categorical_embed_dim

        print(
            f"Model Init: use_text={self.use_text}, use_continuous={self.use_continuous}, use_categorical={self.use_categorical}")
        print(f"Total input dim to MLP: {total_input_dim}")

        if total_input_dim == 0:
            raise ValueError("No features are enabled. Model cannot be built.")

        # --- 4. Classifier Head (MLP) ---
        layer_dims = [total_input_dim] + mlp_config.mlp_hidden_layers

        mlp_layers = []
        for i in range(len(layer_dims) - 1):
            mlp_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
            mlp_layers.append(nn.Dropout(mlp_config.dropout_rate))

        mlp_layers.append(nn.Linear(layer_dims[-1], 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self,
                x_text: torch.Tensor,
                x_continuous: torch.Tensor,
                x_categorical: torch.Tensor) -> torch.Tensor:

        active_features = []

        if self.use_text:
            active_features.append(x_text)

        if self.use_continuous:
            active_features.append(x_continuous)

        if self.use_categorical:
            embedded_cats = []
            for i, name in enumerate(self.categorical_feature_names):
                cat_ids = x_categorical[:, i]
                embedded_cats.append(self.embedding_layers[name](cat_ids))

            all_embedded_cats = torch.cat(embedded_cats, dim=1)
            active_features.append(all_embedded_cats)

        combined_features = torch.cat(active_features, dim=1)
        logits = self.mlp(combined_features)

        return logits.squeeze(-1)