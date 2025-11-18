from dataclasses import dataclass, field
import torch
import torch.nn as nn
from feature_processor import FeatureHyperParams


class HybridModel(nn.Module):

    @dataclass(frozen=True)
    class MlpHyperParams:
        """Holds all hyperparameters for the MLP classifier head."""
        mlp_hidden_layers: list[int] = field(default_factory=lambda: [128, 64])
        dropout_rate: float = 0.3
        text_projection_dim: int | None = None  # e.g., 128

    def __init__(
        self,
        feature_config: FeatureHyperParams,
        mlp_config: MlpHyperParams
    ):

        super().__init__()

        # --- Store which features are active based on the config ---
        self.use_text = feature_config.text_embed_dim > 0
        self.use_continuous = feature_config.continuous_feat_dim > 0
        self.use_categorical = len(feature_config.categorical_vocab_sizes) > 0

        self.categorical_feature_names = list(feature_config.categorical_vocab_sizes.keys())

        total_input_dim = 0

        # --- 1. Text Features ---
        self.text_projector = None
        if self.use_text:
            # Check if we are using a projection layer
            if mlp_config.text_projection_dim and mlp_config.text_projection_dim > 0:
                self.text_projector = nn.Linear(
                    feature_config.text_embed_dim,
                    mlp_config.text_projection_dim
                )
                total_input_dim += mlp_config.text_projection_dim
                print(f"  Using text projection: {feature_config.text_embed_dim} -> {mlp_config.text_projection_dim}")
            else:
                # No projection, use full embedding
                total_input_dim += feature_config.text_embed_dim

        # --- 2. Continuous Features ---
        if self.use_continuous:
            total_input_dim += feature_config.continuous_feat_dim

        # --- 3. Categorical Features ---
        total_categorical_embed_dim = 0
        if self.use_categorical:
            self.embedding_layers = nn.ModuleDict()
            for name, vocab_size in feature_config.categorical_vocab_sizes.items():
                embed_dim = feature_config.embedding_dims.get(name, 16)  # Default dim
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

        self.mlp = nn.Sequential(*mlp_layers)
        self.head = nn.Linear(layer_dims[-1], 1)

    def embed(self,
        x_text: torch.Tensor,
        x_continuous: torch.Tensor,
        x_categorical: torch.Tensor
    ) -> torch.Tensor:

        active_features = []

        if self.use_text:
            if self.text_projector:
                x_text_processed = self.text_projector(x_text)
            else:
                x_text_processed = x_text
            active_features.append(x_text_processed)

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
        return self.mlp(combined_features)

    def forward(self,
        x_text: torch.Tensor,
        x_continuous: torch.Tensor,
        x_categorical: torch.Tensor
    ) -> torch.Tensor:
        embedding = self.embed(x_text, x_continuous, x_categorical)
        logits = self.head(embedding)
        return logits.squeeze(-1)