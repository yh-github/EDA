from common.embedder import EmbeddingService
import torch.nn as nn
from transformers import AutoModel
from .config import MultiExpConfig


class TransactionEncoder(nn.Module):
    def __init__(self, config: MultiExpConfig):
        super().__init__()
        # 1. Text Encoder
        self.embedder = AutoModel.from_pretrained(config.text_encoder_model)
        text_dim = self.embedder.config.hidden_size

        # 2. Feature Projection
        self.text_proj = nn.Linear(text_dim, config.hidden_dim)
        self.amount_proj = nn.Linear(1, config.hidden_dim)
        self.day_proj = nn.Linear(1, config.hidden_dim)

        # 3. Fusion Layer
        self.layer_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, input_ids, attention_mask, amounts, days):
        b, n, seq_len = input_ids.shape
        flat_input_ids = input_ids.view(b * n, seq_len)
        flat_mask = attention_mask.view(b * n, seq_len)

        bert_out = self.embedder(flat_input_ids, attention_mask=flat_mask).last_hidden_state[:, 0, :]

        text_emb = self.text_proj(bert_out).view(b, n, -1)
        amt_emb = self.amount_proj(amounts)
        day_emb = self.day_proj(days)

        fused = text_emb + amt_emb + day_emb
        return self.layer_norm(fused)


class TransactionTransformer(nn.Module):
    def __init__(self, config: MultiExpConfig):
        super().__init__()
        self.encoder = TransactionEncoder(config)

        # Main Transformer (Contextualizer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Head 1: Pairwise Link Prediction (Clustering)
        self.bilinear = nn.Bilinear(config.hidden_dim, config.hidden_dim, 1)

        # Head 2: Cycle Classification (Node Classification)
        self.cycle_head = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, batch):
        x = self.encoder(
            batch['input_ids'],
            batch['attention_mask'],
            batch['amounts'],
            batch['days']
        )

        src_key_padding_mask = ~batch['padding_mask']
        h = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        h_i = h.unsqueeze(2).expand(-1, -1, h.size(1), -1)
        h_j = h.unsqueeze(1).expand(-1, h.size(1), -1, -1)

        adj_logits = self.bilinear(h_i, h_j).squeeze(-1)
        cycle_logits = self.cycle_head(h)

        return adj_logits, cycle_logits