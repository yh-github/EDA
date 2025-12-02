import logging
import torch
import torch.nn as nn
from multi.config import MultiExpConfig
from multi.encoder import TransactionEncoder

logger = logging.getLogger(__name__)


class BinaryTransactionTransformer(nn.Module):
    """
    New Binary Model.
    Predicts:
      1. Adjacency (Clustering)
      2. Binary isRecurring (Detection)
    """

    def __init__(self, config: MultiExpConfig):
        super().__init__()
        self.encoder = TransactionEncoder(config)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Adjacency Head (Same as before)
        self.adj_weight = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.adj_bias = nn.Parameter(torch.zeros(1))

        # Binary Head (Simple Linear Projection)
        # We output 1 dimension -> Logit for P(isRecurring=True)
        self.binary_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Unwrap the correct inputs based on caching strategy
        # The collate_fn ensures the right keys exist (e.g., 'cached_text' vs 'input_ids')

        x = self.encoder(
            input_ids=batch.get('input_ids'),
            attention_mask=batch.get('attention_mask'),
            amounts=batch['amounts'],
            days=batch['days'],
            calendar_features=batch['calendar_features'],
            cp_input_ids=batch.get('cp_input_ids'),
            cp_attention_mask=batch.get('cp_attention_mask'),
            precomputed_text_embs=batch.get('cached_text'),
            precomputed_cp_embs=batch.get('cached_cp')
        )

        src_key_padding_mask = ~batch['padding_mask']
        h = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # 1. Adjacency
        h_transformed = self.adj_weight(h)
        adj_logits = torch.matmul(h_transformed, h.transpose(1, 2)) + self.adj_bias
        adj_logits = (adj_logits + adj_logits.transpose(1, 2)) / 2

        # 2. Binary Detection
        # [B, S, H] -> [B, S, 1]
        binary_logits = self.binary_head(h)

        return adj_logits, binary_logits, h