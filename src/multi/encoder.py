import torch
import torch.nn as nn
from transformers import AutoModel
from multi.config import MultiExpConfig


class TransactionEncoder(nn.Module):
    def __init__(self, config: MultiExpConfig):
        super().__init__()
        self.use_cp = config.use_counter_party

        self.embedder = AutoModel.from_pretrained(config.text_encoder_model)
        text_dim = self.embedder.config.hidden_size

        self.text_proj = nn.Linear(text_dim, config.hidden_dim)

        if self.use_cp:
            self.cp_proj = nn.Linear(text_dim, config.hidden_dim)
        else:
            self.cp_proj = None

        self.amount_proj = nn.Linear(1, config.hidden_dim)
        self.day_proj = nn.Linear(1, config.hidden_dim)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)

    def _encode_text_stream(self, input_ids, attention_mask, projector):
        b, n, seq_len = input_ids.shape
        flat_ids = input_ids.view(b * n, seq_len)
        flat_mask = attention_mask.view(b * n, seq_len)

        bert_out = self.embedder(flat_ids, attention_mask=flat_mask).last_hidden_state[:, 0, :]
        return projector(bert_out).view(b, n, -1)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            amounts: torch.Tensor,
            days: torch.Tensor,
            cp_input_ids: torch.Tensor = None,
            cp_attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Fuses Description + (Optional) CounterParty + Amount + Time.
        """
        desc_emb = self._encode_text_stream(input_ids, attention_mask, self.text_proj)

        amt_emb = self.amount_proj(amounts)
        day_emb = self.day_proj(days)

        fused = desc_emb + amt_emb + day_emb

        if self.use_cp and cp_input_ids is not None:
            cp_emb = self._encode_text_stream(cp_input_ids, cp_attention_mask, self.cp_proj)
            fused = fused + cp_emb

        return self.layer_norm(fused)


class TransactionTransformer(nn.Module):
    def __init__(self, config: MultiExpConfig):
        super().__init__()
        self.encoder = TransactionEncoder(config)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.bilinear = nn.Bilinear(config.hidden_dim, config.hidden_dim, 1)
        self.cycle_head = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(
            batch['input_ids'],
            batch['attention_mask'],
            batch['amounts'],
            batch['days'],
            cp_input_ids=batch.get('cp_input_ids'),
            cp_attention_mask=batch.get('cp_attention_mask')
        )

        src_key_padding_mask = ~batch['padding_mask']
        h = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        h_i = h.unsqueeze(2).expand(-1, -1, h.size(1), -1)
        h_j = h.unsqueeze(1).expand(-1, h.size(1), -1, -1)

        adj_logits = self.bilinear(h_i, h_j).squeeze(-1)
        cycle_logits = self.cycle_head(h)

        return adj_logits, cycle_logits