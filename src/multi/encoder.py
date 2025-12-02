import math
import logging
import torch
import torch.nn as nn
from transformers import AutoModel
from multi.config import MultiExpConfig

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TimeEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(float(max_len)) / hidden_dim))
        self.register_buffer('div_term', div_term)

    def forward(self, days):
        position = days * 1.0
        pe = torch.zeros(days.shape[0], days.shape[1], self.hidden_dim, device=days.device)
        pe[:, :, 0::2] = torch.sin(position * self.div_term)
        pe[:, :, 1::2] = torch.cos(position * self.div_term)
        return pe


class TransactionEncoder(nn.Module):
    @staticmethod
    def _pool_mean(last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    @staticmethod
    def _pool_last_token(last_hidden_state, attention_mask):
        seq_lengths = attention_mask.sum(dim=1) - 1
        seq_lengths = seq_lengths.clamp(min=0)
        batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
        return last_hidden_state[batch_indices, seq_lengths]

    @staticmethod
    def inspect_and_configure_model(model_name: str):
        from transformers import AutoConfig
        try:
            config = AutoConfig.from_pretrained(model_name)
        except OSError:
            return "unknown", "mean"
        arch = config.architectures[0].lower() if config.architectures else ""
        if any(x in arch for x in ["gpt", "opt", "bloom", "llama", "causal"]):
            return "decoder", TransactionEncoder._pool_last_token
        elif any(x in arch for x in ["t5", "bart"]):
            return "encoder_decoder", TransactionEncoder._pool_mean
        else:
            return "encoder", TransactionEncoder._pool_mean

    @staticmethod
    def get_embedder(config: MultiExpConfig):
        embedder = AutoModel.from_pretrained(config.text_encoder_model)
        model_type, pooling_strategy = TransactionEncoder.inspect_and_configure_model(config.text_encoder_model)
        logger.info(f"Model Architecture detected: {model_type}. Using pooling: {pooling_strategy}")

        for param in embedder.parameters():
            param.requires_grad = False

        if config.unfreeze_last_n_layers > 0:
            base_model = getattr(embedder, embedder.base_model_prefix, embedder)
            layers = None
            for attr in ['encoder', 'decoder', 'transformer', 'layers', 'h', 'blocks']:
                if hasattr(base_model, attr):
                    potential_layers = getattr(base_model, attr)
                    if isinstance(potential_layers, (nn.ModuleList, nn.Sequential)) or (
                    hasattr(potential_layers, 'layer')):
                        layers = potential_layers.layer if hasattr(potential_layers, 'layer') else potential_layers
                        break
            if layers is not None:
                logger.info(f"Unfreezing last {config.unfreeze_last_n_layers} layers.")
                for layer in layers[-config.unfreeze_last_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
            else:
                logger.warning(f"Could not automatically locate layers. Keeping fully frozen.")
        return embedder, pooling_strategy

    def __init__(self, config: MultiExpConfig):
        super().__init__()
        self.use_cp = config.use_counter_party
        self.chunk_size = config.chunk_size
        self.embedder, self.pool_text_fn = self.get_embedder(config)
        text_dim = self.embedder.config.hidden_size

        self.text_proj = nn.Linear(text_dim, config.hidden_dim)
        if self.use_cp:
            self.cp_proj = nn.Linear(text_dim, config.hidden_dim)
        else:
            self.cp_proj = None

        self.amount_proj = nn.Linear(1, config.hidden_dim)
        self.time_encoder = TimeEncoding(config.hidden_dim, max_len=config.time_encoding_max_len)

        # 14-day cycle support (Input 6)
        self.calendar_proj = nn.Linear(6, config.hidden_dim)

        if config.normalization_type == 'rms_norm':
            self.layer_norm = RMSNorm(config.hidden_dim)
        elif config.normalization_type == 'layer_norm':
            self.layer_norm = nn.LayerNorm(config.hidden_dim)
        else:
            self.layer_norm = nn.Identity()

    def _encode_text_stream(self, input_ids, attention_mask, projector):
        b, n, seq_len = input_ids.shape
        total_seqs = b * n
        flat_ids = input_ids.view(total_seqs, seq_len)
        flat_mask = attention_mask.view(total_seqs, seq_len)
        chunk_size = self.chunk_size
        embeddings = []
        for i in range(0, total_seqs, chunk_size):
            chunk_ids = flat_ids[i: i + chunk_size]
            chunk_mask = flat_mask[i: i + chunk_size]
            outputs = self.embedder(chunk_ids, attention_mask=chunk_mask)
            pooled = self.pool_text_fn(outputs.last_hidden_state, chunk_mask)
            embeddings.append(projector(pooled))
        full_embedding = torch.cat(embeddings, dim=0)
        return full_embedding.view(b, n, -1)

    def forward(self, input_ids, attention_mask, amounts, days, calendar_features, cp_input_ids=None,
                cp_attention_mask=None):
        desc_emb = self._encode_text_stream(input_ids, attention_mask, self.text_proj)
        amt_emb = self.amount_proj(amounts)
        freq_emb = self.time_encoder(days)
        phase_emb = self.calendar_proj(calendar_features)
        fused = desc_emb + amt_emb + freq_emb + phase_emb
        if self.use_cp and cp_input_ids is not None:
            cp_emb = self._encode_text_stream(cp_input_ids, cp_attention_mask, self.cp_proj)
            fused = fused + cp_emb
        return self.layer_norm(fused)


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

        # 1. Adjacency
        h_transformed = self.adj_weight(h)
        adj_logits = torch.matmul(h_transformed, h.transpose(1, 2)) + self.adj_bias
        adj_logits = (adj_logits + adj_logits.transpose(1, 2)) / 2

        # 2. Binary Detection
        # [B, S, H] -> [B, S, 1]
        binary_logits = self.binary_head(h)

        return adj_logits, binary_logits, h