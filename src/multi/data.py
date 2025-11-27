import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from multi.config import MultiExpConfig


class MultiTransactionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: MultiExpConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        # Safe handling of missing target columns for Inference
        if 'patternCycle' not in df.columns:
            df['patternCycle'] = 'None'
        if 'patternId' not in df.columns:
            df['patternId'] = -1

        # 1. Filter out long cycles (if they exist)
        df = df[~df['patternCycle'].isin(['Annual', 'SemiAnnual'])].copy()

        # 2. Encode Pattern IDs (Extracted to method for cleanliness)
        df = self._encode_pattern_ids(df)

        # 3. Create Direction Column
        df['direction'] = np.sign(df['amount'])
        df = df[df['direction'] != 0]

        # 4. Group by Account AND Direction
        self.groups = [group for _, group in df.groupby(['accountId', 'direction'])]

        # Only shuffle if training
        np.random.shuffle(self.groups)

    def _encode_pattern_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Robustly encodes string/mixed pattern IDs into integers for PyTorch.
        Preserves -1 as the 'Noise' label.
        """
        # Ensure it's a string to handle mixed types safely
        df['patternId_str'] = df['patternId'].astype(str)

        # Factorize creates a unique int for every unique string.
        codes, _ = pd.factorize(df['patternId_str'])
        df['patternId_encoded'] = codes

        # Identify which code corresponds to our "Noise" labels ("-1", "None", "nan")
        # We map these back to -1 so the collate_fn knows to ignore them.
        noise_mask = df['patternId_str'].isin(['-1', '-1.0', 'None', 'nan', '<NA>'])
        df.loc[noise_mask, 'patternId_encoded'] = -1

        return df

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]

        # Cap sequence length
        if len(group) > self.config.max_seq_len:
            group = group.sort_values('date', ascending=True).iloc[-self.config.max_seq_len:]

        # Features
        texts = (group['bankRawDescription'].fillna('') + " " + group['counter_party'].fillna('')).tolist()
        amounts = group['amount'].values.astype(np.float32)
        log_amounts = np.log1p(np.abs(amounts)) * np.sign(amounts)
        dates = pd.to_datetime(group['date'])
        days_since_start = (dates - dates.min()).dt.days.values.astype(np.float32)

        # Targets
        # Use the encoded integer ID we created
        pattern_ids = group['patternId_encoded'].values.astype(np.int64)
        cycles = group['patternCycle'].map(self.config.cycle_map).fillna(0).values.astype(np.int64)

        return {
            "texts": texts,
            "amounts": log_amounts,
            "days": days_since_start,
            "pattern_ids": pattern_ids,
            "cycles": cycles
        }


def collate_fn(batch: list[dict], tokenizer, config: MultiExpConfig):
    """Standard dynamic padding collate function."""

    # Flatten texts
    all_texts = [t for item in batch for t in item['texts']]
    account_lengths = [len(item['texts']) for item in batch]
    max_len_in_batch = max(account_lengths)

    # Tokenize
    encodings = tokenizer(
        all_texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    batch_size = len(batch)
    seq_len = encodings['input_ids'].shape[1]  # Renamed from text_emb_dim

    # Initialize Tensors
    # Shape is (Batch, N_Txns, Seq_Len)
    batched_input_ids = torch.zeros((batch_size, max_len_in_batch, seq_len), dtype=torch.long)
    batched_attention_mask = torch.zeros((batch_size, max_len_in_batch, seq_len), dtype=torch.long)

    batched_amounts = torch.zeros((batch_size, max_len_in_batch, 1), dtype=torch.float32)
    batched_days = torch.zeros((batch_size, max_len_in_batch, 1), dtype=torch.float32)
    batched_cycles = torch.zeros((batch_size, max_len_in_batch), dtype=torch.long)
    batched_adjacency = torch.zeros((batch_size, max_len_in_batch, max_len_in_batch), dtype=torch.float32)
    padding_mask = torch.zeros((batch_size, max_len_in_batch), dtype=torch.bool)

    current_idx = 0
    for i, (item, length) in enumerate(zip(batch, account_lengths)):
        batched_input_ids[i, :length] = encodings['input_ids'][current_idx: current_idx + length]
        batched_attention_mask[i, :length] = encodings['attention_mask'][current_idx: current_idx + length]
        batched_amounts[i, :length, 0] = torch.tensor(item['amounts'])
        batched_days[i, :length, 0] = torch.tensor(item['days'])
        batched_cycles[i, :length] = torch.tensor(item['cycles'])

        # Adjacency Matrix Construction
        # Only construct adjacency if we have valid pattern IDs (not all -1)
        p_ids = item['pattern_ids']
        if np.any(p_ids != -1):
            p_ids_tensor = torch.tensor(p_ids).unsqueeze(1)
            # Match if IDs are same AND ID is not -1
            # Broadcast to create N x N matrix
            matches = (p_ids_tensor == p_ids_tensor.T) & (p_ids_tensor != -1)
            batched_adjacency[i, :length, :length] = matches.float()

        padding_mask[i, :length] = True
        current_idx += length

    return {
        "input_ids": batched_input_ids,
        "attention_mask": batched_attention_mask,
        "amounts": batched_amounts,
        "days": batched_days,
        "adjacency_target": batched_adjacency,
        "cycle_target": batched_cycles,
        "padding_mask": padding_mask
    }


def get_dataloader(df, config: MultiExpConfig, shuffle=True):
    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_model)
    ds = MultiTransactionDataset(df, config, tokenizer)
    return DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: collate_fn(x, tokenizer, config)
    )