import logging
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from feature_processor import FeatureHyperParams

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransformerHyperParams:
    """Holds all hyperparameters for the TabularTransformerModel."""
    # The internal embedding dimension for all features
    d_model: int = 128
    # Number of heads in the MultiHeadAttention
    n_head: int = 4
    # Number of TransformerEncoder layers to stack
    num_encoder_layers: int = 2
    # Hidden layers for the *final* classifier head
    final_mlp_layers: list[int] = field(default_factory=lambda: [64])
    dropout_rate: float = 0.2
    pooling_strategy: str = "cls"  # "cls", "mean", or "max"
    norm_first: bool = False


class TabularTransformerModel(nn.Module):
    """
    A Transformer-based model for our hybrid data.

    It embeds all features (text, continuous, categorical) into a
    common dimension (d_model), treats them as a sequence of tokens,
    and passes them through a TransformerEncoder.
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

        self.feature_embedders = nn.ModuleDict()

        self.pooling_strategy = transformer_config.pooling_strategy
        if self.pooling_strategy not in ["cls", "mean", "max"]:
            raise ValueError(f"Unknown pooling_strategy: {self.pooling_strategy}")

        # A. Text Feature
        if feature_config.text_embed_dim > 0:
            self.feature_embedders['text'] = nn.Linear(
                feature_config.text_embed_dim, d_model
            )

        # B. Continuous Features
        if feature_config.continuous_feat_dim > 0:
            self.feature_embedders['continuous_block'] = nn.Linear(
                feature_config.continuous_feat_dim, d_model
            )

        # C. Categorical Features
        for name, vocab_size in feature_config.categorical_vocab_sizes.items():
            embed_dim = feature_config.embedding_dims.get(name, 16)
            self.feature_embedders[name] = nn.Sequential(
                nn.Embedding(vocab_size, embed_dim),
                nn.Linear(embed_dim, d_model)
            )

        # --- 2. The Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_config.n_head,
            dropout=transformer_config.dropout_rate,
            batch_first=True,  # We use (Batch, Seq, Features)
            norm_first=transformer_config.norm_first
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_config.num_encoder_layers,
            enable_nested_tensor=not transformer_config.norm_first # to prevent warning
        )

        # --- 3. The [CLS] Token ---
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # --- 4. The Final Classifier Head ---
        mlp_head_layers = [d_model] + transformer_config.final_mlp_layers
        mlp_layers = []
        for i in range(len(mlp_head_layers) - 1):
            mlp_layers.append(nn.Linear(mlp_head_layers[i], mlp_head_layers[i + 1]))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(transformer_config.dropout_rate))

        # Final projection to 1 logit
        mlp_layers.append(nn.Linear(mlp_head_layers[-1], 1))

        self.mlp_head = nn.Sequential(*mlp_layers)

        logger.info(f"TabularTransformerModel Initialized. d_model={d_model}, "
                    f"pooling={self.pooling_strategy}, norm_first={transformer_config.norm_first}")

    def get_representation(self,
                           x_text: torch.Tensor,
                           x_continuous: torch.Tensor,
                           x_categorical: torch.Tensor) -> torch.Tensor:
        """
        Passes inputs through the Transformer and Pooling layers,
        returning the dense vector representation (embedding) of size d_model.
        """
        embedded_tokens = []
        batch_size = None

        # 1. Embed Text
        if self.feature_config.text_embed_dim > 0:
            # (Batch, 1, d_model)
            token = self.feature_embedders['text'](x_text).unsqueeze(1)
            embedded_tokens.append(token)
            batch_size = x_text.shape[0]

        # 2. Embed Continuous
        if self.feature_config.continuous_feat_dim > 0:
            token = self.feature_embedders['continuous_block'](x_continuous).unsqueeze(1)
            embedded_tokens.append(token)
            if batch_size is None:
                batch_size = x_continuous.shape[0]

        # 3. Embed Categorical
        if len(self.categorical_names) > 0:
            # If this is the first active feature, set batch_size
            if batch_size is None:
                batch_size = x_categorical.shape[0]

            for i, name in enumerate(self.categorical_names):
                cat_token = x_categorical[:, i]
                token = self.feature_embedders[name](cat_token).unsqueeze(1)
                embedded_tokens.append(token)

        if batch_size is None:
            raise ValueError("Model has no inputs enabled.")

        # 4. Prepend CLS token (if using CLS strategy)
        if self.pooling_strategy == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            all_tokens = [cls_tokens] + embedded_tokens
        else:
            all_tokens = embedded_tokens

        # 5. Concatenate into a sequence: (B, num_tokens, d_model)
        feature_sequence = torch.cat(all_tokens, dim=1)

        # 6. Pass through Transformer
        transformer_output = self.transformer_encoder(feature_sequence)

        # 7. Pooling
        if self.pooling_strategy == "cls":
            pooled_output = transformer_output[:, 0]
        elif self.pooling_strategy == "mean":
            pooled_output = transformer_output.mean(dim=1)
        elif self.pooling_strategy == "max":
            pooled_output = torch.max(transformer_output, dim=1).values
        else:
            raise RuntimeError(f"Invalid pooling_strategy '{self.pooling_strategy}'")

        return pooled_output

    def forward(self,
                x_text: torch.Tensor,
                x_continuous: torch.Tensor,
                x_categorical: torch.Tensor) -> torch.Tensor:

        # 1. Get the learned representation
        embedding = self.get_representation(x_text, x_continuous, x_categorical)

        # 2. Classify
        logits = self.mlp_head(embedding)

        return logits.squeeze(-1)