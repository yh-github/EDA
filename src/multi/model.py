import logging
import torch
import torch.nn as nn
from multi.config import MultiExpConfig
from multi.encoder import TransactionEncoder

logger = logging.getLogger(__name__)


class TransactionTransformer(nn.Module):
    """
    Original Multiclass Model.
    """

    def __init__(self, config: MultiExpConfig):
        super().__init__()
        self.encoder = TransactionEncoder(config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim, nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4, dropout=config.dropout,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.adj_weight = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.adj_bias = nn.Parameter(torch.zeros(1))

        self.edge_informed_type = getattr(config, 'edge_informed_type', None)
        cycle_input_dim = config.hidden_dim
        if self.edge_informed_type == "edge_informed_max":
            cycle_input_dim += 1

        self.cycle_head = nn.Linear(cycle_input_dim, config.num_classes)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder(
            batch['input_ids'], batch['attention_mask'], batch['amounts'], batch['days'], batch['calendar_features'],
            cp_input_ids=batch.get('cp_input_ids'), cp_attention_mask=batch.get('cp_attention_mask')
        )
        src_key_padding_mask = ~batch['padding_mask']
        h = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        h_transformed = self.adj_weight(h)
        adj_logits = torch.matmul(h_transformed, h.transpose(1, 2)) + self.adj_bias
        adj_logits = (adj_logits + adj_logits.transpose(1, 2)) / 2

        if self.edge_informed_type == "edge_informed_max":
            B, N, _ = adj_logits.shape
            eye = torch.eye(N, device=adj_logits.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
            padding_mask = batch['padding_mask']
            valid_col_mask = padding_mask.unsqueeze(1)
            mask_to_fill = eye | (~valid_col_mask)
            adj_masked = adj_logits.masked_fill(mask_to_fill, -100.0)  # Safe negative
            max_score, _ = adj_masked.max(dim=2)
            h_augmented = torch.cat([h, max_score.unsqueeze(-1)], dim=-1)
            cycle_logits = self.cycle_head(h_augmented)
        else:
            cycle_logits = self.cycle_head(h)

        return adj_logits, cycle_logits, h

