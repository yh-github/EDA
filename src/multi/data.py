import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from multi.config import MultiExpConfig, MultiFieldConfig

logger = logging.getLogger(__name__)


class MultiTransactionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: MultiExpConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.fields = MultiFieldConfig() # Use centralized field config

        # Safe handling of missing target columns for Inference
        if self.fields.patternCycle not in df.columns:
            df[self.fields.patternCycle] = 'None'
        if self.fields.patternId not in df.columns:
            df[self.fields.patternId] = -1

        # 1. Normalize 'patternCycle' to handle NaNs as 'None' (Non-recurring)
        df[self.fields.patternCycle] = df[self.fields.patternCycle].fillna('None')

        # 2. Filter out any cycle NOT in the config map
        valid_cycles = set(config.cycle_map.keys())
        mask = df[self.fields.patternCycle].isin(valid_cycles)

        if not mask.all():
            removed_df = df[~mask]
            counts = removed_df[self.fields.patternCycle].value_counts().to_dict()
            logger.info(f"Filtering out {len(removed_df)} rows with unknown cycles: {counts}")
            df = df[mask].copy()

        # 3. Encode Pattern IDs
        df = self._encode_pattern_ids(df)

        # 4. Create Direction Column
        # Filter 0 amounts
        df['direction'] = np.sign(df[self.fields.amount])
        df = df[df['direction'] != 0]

        # 5. Group by Account AND Direction
        self.groups = [group for _, group in df.groupby([self.fields.accountId, 'direction'])]

    def _encode_pattern_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Robustly encodes string/mixed pattern IDs into integers for PyTorch.
        Preserves -1 as the 'Noise' label.
        """
        col = self.fields.patternId
        # Ensure it's a string to handle mixed types safely
        df['patternId_str'] = df[col].astype(str)

        # Factorize creates a unique int for every unique string.
        codes, _ = pd.factorize(df['patternId_str'])
        df['patternId_encoded'] = codes

        # Identify which code corresponds to our "Noise" labels
        noise_mask = df['patternId_str'].isin(['-1', '-1.0', 'None', 'nan', '<NA>'])
        df.loc[noise_mask, 'patternId_encoded'] = -1

        return df

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        f = self.fields

        # Cap sequence length (taking the most recent transactions)
        if len(group) > self.config.max_seq_len:
            group = group.sort_values(f.date, ascending=True).iloc[-self.config.max_seq_len:]

        # Features
        # FIX: Handle missing counter_party gracefully if it doesn't exist in DataFrame
        desc = group[f.text].fillna('')
        if f.counter_party in group.columns:
            cp = group[f.counter_party].fillna('')
            texts = (desc + " " + cp).tolist()
        else:
            texts = desc.tolist()

        amounts = group[f.amount].values.astype(np.float32)
        log_amounts = np.log1p(np.abs(amounts)) * np.sign(amounts)
        dates = pd.to_datetime(group[f.date])
        days_since_start = (dates - dates.min()).dt.days.values.astype(np.float32)

        # Targets
        pattern_ids = group['patternId_encoded'].values.astype(np.int64)
        cycles = group[f.patternCycle].map(self.config.cycle_map).fillna(0).values.astype(np.int64)

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
    seq_len = encodings['input_ids'].shape[1]

    # Initialize Tensors
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
        p_ids = item['pattern_ids']
        # Only check adjacency if we have valid patterns (not all -1)
        if np.any(p_ids != -1):
            p_ids_tensor = torch.tensor(p_ids).unsqueeze(1)
            # Match if IDs are same AND ID is not -1
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
        shuffle=shuffle, # This now controls the shuffling correctly
        collate_fn=lambda x: collate_fn(x, tokenizer, config)
    )