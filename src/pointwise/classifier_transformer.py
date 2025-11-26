import logging
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from common.feature_processor import FeatureHyperParams

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransformerHyperParams:
    """Holds all hyperparameters for the TabularTransformerModel."""
    d_model: int = 128
    n_head: int = 4
    num_encoder_layers: int = 2
    final_mlp_layers: list[int] = field(default_factory=lambda: [64])
    dropout_rate: float = 0.2
    pooling_strategy: str = "cls"  # "cls", "mean", or "max"
    norm_first: bool = False


class TabularTransformerModel(nn.Module):
    """
    A Refactored FT-Transformer (Feature Tokenizer Transformer).

    FIXES APPLIED:
    1. Continuous features are now embedded individually (1 token per feature).
    2. Added learnable Feature Embeddings so the model knows which column is which.
    """

    def __init__(
            self,
            feature_config: FeatureHyperParams,
            transformer_config: TransformerHyperParams
    ):
        super().__init__()

        d_model = transformer_config.d_model
        self.feature_config = feature_config
        self.categorical_names = list(feature_config.categorical_vocab_sizes.keys())
        self.pooling_strategy = transformer_config.pooling_strategy

        if self.pooling_strategy not in ["cls", "mean", "max"]:
            raise ValueError(f"Unknown pooling_strategy: {self.pooling_strategy}")

        # --- 1. Feature Tokenizers (Embeddings) ---
        self.feature_embedders = nn.ModuleDict()

        # A. Text Feature (1 Token)
        self.use_text = feature_config.text_embed_dim > 0
        if self.use_text:
            self.feature_embedders['text'] = nn.Linear(
                feature_config.text_embed_dim, d_model
            )

        # B. Continuous Features (N Tokens - FIX: One Linear layer per feature)
        self.num_continuous = feature_config.continuous_feat_dim
        if self.num_continuous > 0:
            # We use a ModuleList to have separate weights for 'amount' vs 'date_sin' etc.
            self.continuous_projectors = nn.ModuleList([
                nn.Linear(1, d_model) for _ in range(self.num_continuous)
            ])

        # C. Categorical Features (M Tokens)
        for name, vocab_size in feature_config.categorical_vocab_sizes.items():
            embed_dim = feature_config.embedding_dims.get(name, 16)
            # We project distinct embedding dims up to d_model
            self.feature_embedders[name] = nn.Sequential(
                nn.Embedding(vocab_size, embed_dim),
                nn.Linear(embed_dim, d_model)
            )

        # --- 2. Feature/Column Embeddings (FIX: The "Positional" Encoding) ---
        # Total tokens = [CLS] + (Text?) + (Cont_1...Cont_N) + (Cat_1...Cat_M)
        num_tokens = 1  # CLS
        if self.use_text: num_tokens += 1
        num_tokens += self.num_continuous
        num_tokens += len(self.categorical_names)

        self.column_embedding = nn.Parameter(torch.randn(1, num_tokens, d_model))

        # --- 3. The Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_config.n_head,
            dropout=transformer_config.dropout_rate,
            batch_first=True,
            norm_first=transformer_config.norm_first,
            activation="gelu"  # GELU is often better for Transformers
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_config.num_encoder_layers
        )

        # --- 4. The [CLS] Token ---
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # --- 5. Classifier Head ---
        mlp_head_layers = [d_model] + transformer_config.final_mlp_layers
        mlp_layers = []
        for i in range(len(mlp_head_layers) - 1):
            mlp_layers.append(nn.Linear(mlp_head_layers[i], mlp_head_layers[i + 1]))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(transformer_config.dropout_rate))

        mlp_layers.append(nn.Linear(mlp_head_layers[-1], 1))
        self.mlp_head = nn.Sequential(*mlp_layers)

        logger.info(f"TabularTransformerModel Refactored. Tokens per row: {num_tokens}")

    def get_representation(self,
                           x_text: torch.Tensor,
                           x_continuous: torch.Tensor,
                           x_categorical: torch.Tensor) -> torch.Tensor:

        batch_size = x_categorical.shape[0] if len(self.categorical_names) > 0 else x_text.shape[0]
        token_list = []

        # 1. [CLS] Token
        token_list.append(self.cls_token.expand(batch_size, -1, -1))

        # 2. Text Token
        if self.use_text:
            # Project (Batch, 768) -> (Batch, 1, d_model)
            token_list.append(self.feature_embedders['text'](x_text).unsqueeze(1))

        # 3. Continuous Tokens (Iterate columns)
        if self.num_continuous > 0:
            # x_continuous is (Batch, N_features)
            for i in range(self.num_continuous):
                # Slice specific column: (Batch, 1)
                col_val = x_continuous[:, i:i + 1]
                # Project: (Batch, 1) -> (Batch, 1, d_model)
                token_list.append(self.continuous_projectors[i](col_val).unsqueeze(1))

        # 4. Categorical Tokens
        for name in self.categorical_names:
            # Lookup & Project: (Batch) -> (Batch, 1, d_model)
            cat_ids = x_categorical[:, self.categorical_names.index(name)]
            token_list.append(self.feature_embedders[name](cat_ids).unsqueeze(1))

        # 5. Stack into Sequence
        # Shape: (Batch, Num_Tokens, d_model)
        x = torch.cat(token_list, dim=1)

        # 6. Add Column Embeddings (Crucial Fix)
        # Broadcasts (1, Num_Tokens, d_model) across batch
        x = x + self.column_embedding

        # 7. Transformer Pass
        x = self.transformer_encoder(x)

        # 8. Pooling
        if self.pooling_strategy == "cls":
            return x[:, 0]  # Return the transformed [CLS] token
        elif self.pooling_strategy == "mean":
            return x.mean(dim=1)
        elif self.pooling_strategy == "max":
            return torch.max(x, dim=1).values

        return x[:, 0]

    def forward(self,
                x_text: torch.Tensor,
                x_continuous: torch.Tensor,
                x_categorical: torch.Tensor) -> torch.Tensor:
        embedding = self.get_representation(x_text, x_continuous, x_categorical)
        logits = self.mlp_head(embedding)
        return logits.squeeze(-1)