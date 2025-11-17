from dataclasses import dataclass, field
import torch
import torch.nn as nn
from feature_processor import FeatureHyperParams


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
    pooling_strategy: str = "cls"


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
        num_features = 0  # Count how many "tokens" we will have

        self.pooling_strategy = transformer_config.pooling_strategy
        if self.pooling_strategy not in ["cls", "mean", "max"]:
            raise ValueError(f"Unknown pooling_strategy: {self.pooling_strategy}")

        # A. Text Feature
        if feature_config.text_embed_dim > 0:
            self.feature_embedders['text'] = nn.Linear(
                feature_config.text_embed_dim, d_model
            )
            num_features += 1

        # B. Continuous Features (Revised)
        if feature_config.continuous_feat_dim > 0:
            self.feature_embedders['continuous_block'] = nn.Linear(
                feature_config.continuous_feat_dim, d_model
            )
            num_features += 1

        # C. Categorical Features
        for name, vocab_size in feature_config.categorical_vocab_sizes.items():
            embed_dim = feature_config.embedding_dims.get(name, 16)
            self.feature_embedders[name] = nn.Sequential(
                nn.Embedding(vocab_size, embed_dim),
                nn.Linear(embed_dim, d_model)
            )
        num_features += len(self.categorical_names)

        # --- 2. The Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_config.n_head,
            dropout=transformer_config.dropout_rate,
            batch_first=True  # We use (Batch, Seq, Features)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_config.num_encoder_layers
        )

        # --- 3. The [CLS] Token ---
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        num_features += 1  # Add one for the CLS token

        # --- 4. The Final Classifier Head ---
        mlp_head_layers = [d_model] + transformer_config.final_mlp_layers
        mlp_layers = []
        for i in range(len(mlp_head_layers) - 1):
            mlp_layers.append(nn.Linear(mlp_head_layers[i], mlp_head_layers[i + 1]))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(transformer_config.dropout_rate))
        mlp_layers.append(nn.Linear(mlp_head_layers[-1], 1))
        self.mlp_head = nn.Sequential(*mlp_layers)

        print(f"TabularTransformerModel Initialized. d_model={d_model}, num_tokens={num_features}")

    def forward(self,
                x_text: torch.Tensor,
                x_continuous: torch.Tensor,
                x_categorical: torch.Tensor) -> torch.Tensor:

        # 1. Embed all features into (B, 1, d_model)
        # Use .unsqueeze(1) to create the "sequence" dimension
        embedded_tokens = []

        if self.feature_config.text_embed_dim > 0:
            embedded_tokens.append(self.feature_embedders['text'](x_text).unsqueeze(1))

        if self.feature_config.continuous_feat_dim > 0:
            embedded_tokens.append(
                self.feature_embedders['continuous_block'](x_continuous).unsqueeze(1)
            )

        if len(self.categorical_names) > 0:
            for i, name in enumerate(self.categorical_names):
                cat_token = x_categorical[:, i]  # Get the i-th categorical feature
                embedded_tokens.append(
                    self.feature_embedders[name](cat_token).unsqueeze(1)
                )

        # 2. Prepend CLS token
        # Expand CLS token to match batch size
        batch_size = -1
        if self.feature_config.text_embed_dim > 0:
            batch_size = x_text.shape[0]
        elif self.feature_config.continuous_feat_dim > 0:
            batch_size = x_continuous.shape[0]
        elif len(self.categorical_names) > 0:
            batch_size = x_categorical.shape[0]
        else:
            # This should be impossible if the model built correctly
            raise ValueError("Model has no inputs enabled.")

        if self.pooling_strategy == "cls":
            # (Note: This is the line we discussed for the bug fix,
            # you can apply the more robust batch_size logic here when ready)
            batch_size = x_text.shape[0] if self.feature_config.text_embed_dim > 0 else x_continuous.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            all_tokens = [cls_tokens] + embedded_tokens
        else:
            all_tokens = embedded_tokens  # No CLS token


        # 3. Concatenate into a sequence: (B, num_tokens, d_model)
        feature_sequence = torch.cat(all_tokens, dim=1)

        # 4. Pass through Transformer
        # Output shape is also (B, num_tokens, d_model)
        transformer_output = self.transformer_encoder(feature_sequence)

        # 5. Get the output of the [CLS] token (the first token)
        if self.pooling_strategy == "cls":
            # Get the output of the [CLS] token (the first token)
            pooled_output = transformer_output[:, 0]

        elif self.pooling_strategy == "mean":
            # Take the mean of all token outputs
            # dim=1 is the sequence dimension
            pooled_output = transformer_output.mean(dim=1)

        elif self.pooling_strategy == "max":
            # Take the max of all token outputs
            # .max returns (values, indices), we just want the values
            pooled_output = torch.max(transformer_output, dim=1).values

        else:
            # This should be impossible due to the __init__ check
            raise RuntimeError(f"Invalid pooling_strategy '{self.pooling_strategy}' in forward pass.")

        # 6. Classify
        logits = self.mlp_head(pooled_output)

        return logits.squeeze(-1)