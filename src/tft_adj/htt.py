import torch
import torch.nn as nn
import math
from multi.config import MultiExpConfig
from multi.encoder import TransactionEncoder
from common.tft_layers import VariableSelectionNetwork, GatedResidualNetwork


class TimeAwareSelfAttention(nn.Module):
    """
    Self-Attention that injects a bias based on the time difference (days)
    between transactions i and j.
    """

    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        # Time Bias Network
        # Projects a scalar (day_diff) into a bias for each head
        self.time_bias_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, num_heads)  # One bias scalar per head
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, days, padding_mask):
        # x: [Batch, Seq, Hidden]
        # days: [Batch, Seq]
        # padding_mask: [Batch, Seq] (True = padding)

        B, S, H = x.shape

        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, Heads, S, HeadDim]
        K = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # 1. Standard Attention Scores
        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale  # [B, Heads, S, S]

        # 2. Time Relative Bias
        # Calculate pairwise differences: days[i] - days[j]
        days_exp = days.unsqueeze(2)  # [B, S, 1]
        days_T = days.unsqueeze(1)  # [B, 1, S]
        # Use Absolute difference (recurrence is symmetric)
        diff_matrix = torch.abs(days_exp - days_T).unsqueeze(-1)  # [B, S, S, 1]

        # Compute Bias
        # [B, S, S, Heads] -> permute to [B, Heads, S, S]
        time_bias = self.time_bias_net(diff_matrix).permute(0, 3, 1, 2)

        # Inject Bias
        attn_scores = attn_scores + time_bias

        # 3. Masking
        # padding_mask is True for PADDING. We want to mask those out (-inf).
        # Expand mask to [B, 1, 1, S] for broadcasting
        mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2)
        attn_scores = attn_scores.masked_fill(mask_expanded, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 4. Output
        context = (attn_probs @ V).transpose(1, 2).reshape(B, S, H)
        return self.o_proj(context)


class HybridTransactionTransformer(nn.Module):
    def __init__(self, config: MultiExpConfig):
        super().__init__()
        self.config = config

        # --- 1. Feature Encoders ---
        # Re-use the TransactionEncoder to get raw embeddings (Text, CP, etc.)
        # But we won't use its forward() fusion logic. We take the pieces.
        self.raw_encoder = TransactionEncoder(config)

        # We need to know sizes to init VSN
        # Text: HiddenDim (projected inside encoder)
        # Amount: 1 -> HiddenDim
        # Day: HiddenDim (TimeEncoding)
        # Calendar: HiddenDim (Projected)

        self.vsn = VariableSelectionNetwork(
            input_sizes={
                'text': config.hidden_dim,
                'amount': config.hidden_dim,
                'calendar': config.hidden_dim
                # Note: We treat 'Time' specially in the attention,
                # but we can also include absolute time in VSN if we want.
            },
            hidden_size=config.hidden_dim,
            dropout=config.dropout
        )

        # --- 2. Contextualizer (Transformer) ---
        # We replace the standard layer with our TimeAware one
        self.layers = nn.ModuleList([
            TimeAwareSelfAttention(config.hidden_dim, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])

        self.norms = nn.ModuleList([nn.LayerNorm(config.hidden_dim) for _ in range(config.num_layers)])
        self.ffns = nn.ModuleList([
            GatedResidualNetwork(config.hidden_dim, config.hidden_dim, config.hidden_dim, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(config.hidden_dim) for _ in range(config.num_layers)])

        # --- 3. Heads ---
        self.adj_weight = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.adj_bias = nn.Parameter(torch.zeros(1))
        self.binary_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # A. Extract Raw Features (using helper from encoder.py)
        # 1. Text
        if self.config.use_cached_embeddings:
            text_emb = self.raw_encoder.text_proj(batch['cached_text'])
        else:
            text_emb = self.raw_encoder._encode_text_stream(
                batch['input_ids'], batch['attention_mask'], self.raw_encoder.text_proj
            )

        # 2. Amount
        amt_emb = self.raw_encoder.amount_proj(batch['amounts'])

        # 3. Calendar
        cal_emb = self.raw_encoder.calendar_proj(batch['calendar_features'])

        # B. Variable Selection (Fusion)
        # We combine [Text, Amount, Calendar] intelligently
        feature_dict = {
            'text': text_emb,
            'amount': amt_emb,
            'calendar': cal_emb
        }
        x, _ = self.vsn(feature_dict)

        # C. Time-Aware Transformer
        days = batch['days']
        padding_mask = ~batch['padding_mask']  # Invert for masked_fill (True = Mask)

        for i, layer in enumerate(self.layers):
            # Attention Block
            residual = x
            x = layer(x, days, padding_mask)
            x = self.norms[i](x + residual)

            # Feed Forward Block (GRN instead of simple MLP)
            residual = x
            x = self.ffns[i](x)
            x = self.ffn_norms[i](x + residual)

        h = x

        # D. Heads
        # 1. Adjacency
        h_transformed = self.adj_weight(h)
        adj_logits = torch.matmul(h_transformed, h.transpose(1, 2)) + self.adj_bias
        adj_logits = (adj_logits + adj_logits.transpose(1, 2)) / 2

        # 2. Binary
        binary_logits = self.binary_head(h)

        return adj_logits, binary_logits, h