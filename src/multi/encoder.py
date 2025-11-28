import math
import logging
import torch
import torch.nn as nn
from transformers import AutoModel
from multi.config import MultiExpConfig

logger = logging.getLogger(__name__)


class TimeEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for time (days).
    Allows the model to learn periodicity (e.g., 7 days, 30 days).
    """

    def __init__(self, hidden_dim: int, max_len: int = 10000):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Create constant 'pe' matrix with values dependent on pos and i
        # We register it as a buffer so it saves with the model but isn't a trained parameter
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(float(max_len)) / hidden_dim))
        self.register_buffer('div_term', div_term)

    def forward(self, days):
        """
        Args:
            days: Tensor of shape [Batch, Seq, 1] containing day offsets (float)
        """
        # Output shape: [Batch, Seq, Hidden_Dim]
        # We use the scalar 'days' values as the 'position'
        # days is [B, S, 1], div_term is [H/2]
        position = days * 1.0

        pe = torch.zeros(days.shape[0], days.shape[1], self.hidden_dim, device=days.device)
        pe[:, :, 0::2] = torch.sin(position * self.div_term)
        pe[:, :, 1::2] = torch.cos(position * self.div_term)

        return pe


class TransactionEncoder(nn.Module):

    @staticmethod
    def get_embedder(config: MultiExpConfig):
        embedder = AutoModel.from_pretrained(config.text_encoder_model)

        # Logic to handle freezing/unfreezing
        for param in embedder.parameters():
            param.requires_grad = False

        if config.unfreeze_last_n_layers == 0:
            return embedder

        # config.unfreeze_last_n_layers > 0:
        # Freeze everything first (Redundant but safe)
        for param in embedder.parameters():
            param.requires_grad = False

        # Unfreeze the specific top layers (Encoder + Pooler)
        if hasattr(embedder, 'encoder') and hasattr(embedder.encoder, 'layer'):
            # BERT / MPNET / ALBERT
            encoder_layers = embedder.encoder.layer
        elif hasattr(embedder, 'transformer') and hasattr(embedder.transformer, 'layer'):
            # DistilBERT
            encoder_layers = embedder.transformer.layer
        else:
            # Fallback: Just print warning and skip specific layer unfreezing
            print("Warning: Could not locate encoder layers for unfreezing (architecture unknown). Keeping frozen.")
            encoder_layers = []

        for layer in encoder_layers[-config.unfreeze_last_n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Always unfreeze the pooler if it exists
        if hasattr(embedder, 'pooler') and embedder.pooler is not None:
            for param in embedder.pooler.parameters():
                param.requires_grad = True

        try:
            embedder.gradient_checkpointing_enable()
        except ValueError as ve:
            logger.warning(f'{str(ve)}')
        return embedder

    def __init__(self, config: MultiExpConfig):
        super().__init__()
        self.use_cp = config.use_counter_party
        self.chunk_size = config.chunk_size

        self.embedder = self.get_embedder(config)

        text_dim = self.embedder.config.hidden_size

        self.text_proj = nn.Linear(text_dim, config.hidden_dim)

        if self.use_cp:
            self.cp_proj = nn.Linear(text_dim, config.hidden_dim)
        else:
            self.cp_proj = None

        self.amount_proj = nn.Linear(1, config.hidden_dim)
        self.time_encoder = TimeEncoding(config.hidden_dim, max_len=config.time_encoding_max_len)
        self.calendar_proj = nn.Linear(4, config.hidden_dim)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)

    def _encode_text_stream(self, input_ids, attention_mask, projector):
        """
        Encodes text in chunks to save memory.
        """
        b, n, seq_len = input_ids.shape
        total_seqs = b * n

        flat_ids = input_ids.view(total_seqs, seq_len)
        flat_mask = attention_mask.view(total_seqs, seq_len)

        chunk_size = self.chunk_size
        embeddings = []

        for i in range(0, total_seqs, chunk_size):
            chunk_ids = flat_ids[i: i + chunk_size]
            chunk_mask = flat_mask[i: i + chunk_size]

            # Forward pass for chunk
            bert_out = self.embedder(chunk_ids, attention_mask=chunk_mask).last_hidden_state[:, 0, :]
            proj_out = projector(bert_out)
            embeddings.append(proj_out)

        # Concatenate and reshape
        full_embedding = torch.cat(embeddings, dim=0)
        return full_embedding.view(b, n, -1)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            amounts: torch.Tensor,
            days: torch.Tensor,
            calendar_features: torch.Tensor,
            cp_input_ids: torch.Tensor = None,
            cp_attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Fuses Description + (Optional) CounterParty + Amount + Time.
        """
        desc_emb = self._encode_text_stream(input_ids, attention_mask, self.text_proj)

        amt_emb = self.amount_proj(amounts)

        # Helps learn "Every 28 days"
        freq_emb = self.time_encoder(days)

        # 4. Phase Context (Absolute Calendar)
        # Helps learn "On the 1st" or "On weekends"
        phase_emb = self.calendar_proj(calendar_features)

        fused = desc_emb + amt_emb + freq_emb + phase_emb

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
            batch_first=True,
            norm_first=True  # Usually stabilizes training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Replaced Bilinear with explicit weights for memory efficient matmul
        self.adj_weight = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.adj_bias = nn.Parameter(torch.zeros(1))

        self.cycle_head = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder(
            batch['input_ids'],
            batch['attention_mask'],
            batch['amounts'],
            batch['days'],
            batch['calendar_features'],
            cp_input_ids=batch.get('cp_input_ids'),
            cp_attention_mask=batch.get('cp_attention_mask')
        )

        src_key_padding_mask = ~batch['padding_mask']
        h = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Compute Adjacency (Memory Efficient)
        # 1. Project: [B, N, D] @ [D, D] -> [B, N, D]
        h_transformed = self.adj_weight(h)

        # 2. Dot Product: [B, N, D] @ [B, D, N] -> [B, N, N]
        adj_logits = torch.matmul(h_transformed, h.transpose(1, 2)) + self.adj_bias

        # FIX: Enforce Symmetry (A matches B implies B matches A)
        adj_logits = (adj_logits + adj_logits.transpose(1, 2)) / 2

        # Compute Cycles
        cycle_logits = self.cycle_head(h)

        # Return h for Contrastive Loss
        return adj_logits, cycle_logits, h