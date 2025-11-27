import torch
import torch.nn as nn
from transformers import AutoModel
from multi.config import MultiExpConfig


class TransactionEncoder(nn.Module):
    def __init__(self, config: MultiExpConfig):
        super().__init__()
        # 1. Text Encoder
        # We use a pre-trained Transformer (e.g., MiniLM) to extract text features.
        self.embedder = AutoModel.from_pretrained(config.text_encoder_model)
        text_dim = self.embedder.config.hidden_size

        # 2. Feature Projection
        # Project distinct feature types into a shared latent space.
        self.text_proj = nn.Linear(text_dim, config.hidden_dim)
        self.amount_proj = nn.Linear(1, config.hidden_dim)
        self.day_proj = nn.Linear(1, config.hidden_dim)

        # 3. Fusion Layer
        self.layer_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            amounts: torch.Tensor,
            days: torch.Tensor
    ) -> torch.Tensor:
        """
        Encodes raw transaction features into a single fused vector per transaction.

        Args:
            input_ids (torch.Tensor): Token IDs from tokenizer.
                Shape: (Batch_Size, Num_Transactions, Seq_Len)
            attention_mask (torch.Tensor): Attention mask from tokenizer.
                Shape: (Batch_Size, Num_Transactions, Seq_Len)
            amounts (torch.Tensor): Log-normalized, signed amounts.
                Shape: (Batch_Size, Num_Transactions, 1)
            days (torch.Tensor): Normalized days since account start.
                Shape: (Batch_Size, Num_Transactions, 1)

        Returns:
            torch.Tensor: Fused embeddings.
                Shape: (Batch_Size, Num_Transactions, Hidden_Dim)
        """
        # Dimensions: B=Batch, N=Transactions per Account, S=Sequence Length
        b, n, seq_len = input_ids.shape

        # 1. Encode Text
        # Flatten batch and N dims to feed into standard BERT: (B*N, S)
        flat_input_ids = input_ids.view(b * n, seq_len)
        flat_mask = attention_mask.view(b * n, seq_len)

        # Extract [CLS] token (index 0) from last hidden state
        # Output: (B*N, Text_Hidden_Dim)
        bert_out = self.embedder(flat_input_ids, attention_mask=flat_mask).last_hidden_state[:, 0, :]

        # Project and reshape back to (B, N, H)
        text_emb = self.text_proj(bert_out).view(b, n, -1)

        # 2. Encode Scalars
        # amounts: (B, N, 1) -> (B, N, H)
        amt_emb = self.amount_proj(amounts)
        # days: (B, N, 1) -> (B, N, H)
        day_emb = self.day_proj(days)

        # 3. Fuse (Element-wise Sum)
        # Broadcasting handles dimensions automatically
        fused = text_emb + amt_emb + day_emb

        return self.layer_norm(fused)


class TransactionTransformer(nn.Module):
    def __init__(self, config: MultiExpConfig):
        super().__init__()
        self.encoder = TransactionEncoder(config)

        # Main Transformer (Contextualizer)
        # Attends to ALL transactions in the account history simultaneously.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Head 1: Pairwise Link Prediction (Clustering)
        # Computes similarity score between every pair of transactions (N x N)
        self.bilinear = nn.Bilinear(config.hidden_dim, config.hidden_dim, 1)

        # Head 2: Cycle Classification (Node Classification)
        # Predicts cycle label (Monthly, Weekly, None) for EACH transaction
        self.cycle_head = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Args:
            batch (dict): Dictionary from collate_fn containing:
                - input_ids: (B, N, S)
                - attention_mask: (B, N, S)
                - amounts: (B, N, 1)
                - days: (B, N, 1)
                - padding_mask: (B, N) where True = Valid Data, False = Padding

        Returns:
            tuple:
                adj_logits (torch.Tensor): (B, N, N) - Pairwise connection scores
                cycle_logits (torch.Tensor): (B, N, Num_Classes) - Per-node classification
        """
        # 1. Encode Features -> (B, N, H)
        x = self.encoder(
            batch['input_ids'],
            batch['attention_mask'],
            batch['amounts'],
            batch['days']
        )

        # 2. Contextualize via Transformer
        # PyTorch Transformer mask logic: True means "ignore this position" (Padding).
        # Our loader provides 'padding_mask' where True means "Valid Data".
        # So we invert the mask: ~batch['padding_mask'] -> True where it is Padding.
        src_key_padding_mask = ~batch['padding_mask']

        # h shape: (B, N, H)
        h = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # 3. Head 1: Adjacency (Bilinear)
        # Prepare for pairwise comparison (Broadcasting)
        # h_i: Source nodes (B, N, 1, H)
        # h_j: Target nodes (B, 1, N, H)
        h_i = h.unsqueeze(2).expand(-1, -1, h.size(1), -1)
        h_j = h.unsqueeze(1).expand(-1, h.size(1), -1, -1)

        # adj_logits: (B, N, N)
        adj_logits = self.bilinear(h_i, h_j).squeeze(-1)

        # 4. Head 2: Cycle Classification
        # cycle_logits: (B, N, C)
        cycle_logits = self.cycle_head(h)

        return adj_logits, cycle_logits