import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from multi.config import MultiExpConfig


class MultiTransactionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: MultiExpConfig, tokenizer):
        """
        Groups transactions by AccountID AND Direction (Credit/Debit).
        This reduces N and isolates distinct patterns.
        """
        self.config = config
        self.tokenizer = tokenizer

        # 1. Filter out long cycles
        df = df[~df['patternCycle'].isin(['Annual', 'SemiAnnual'])].copy()

        # 2. Fill NaNs
        df['patternId'] = df['patternId'].fillna(-1)
        df['patternCycle'] = df['patternCycle'].fillna('None')

        # 3. Create Direction Column (1 for Credit, -1 for Debit)
        # We handle 0 amounts as Debit or ignore them
        df['direction'] = np.sign(df['amount'])
        # Filter out 0 amounts if they exist (usually failed txns)
        df = df[df['direction'] != 0]

        # 4. Group by Account AND Direction
        # This effectively splits every account into two independent samples
        self.groups = [group for _, group in df.groupby(['accountId', 'direction'])]

        # Shuffle groups to mix credits and debits in batches
        np.random.shuffle(self.groups)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]

        # Cap sequence length
        if len(group) > self.config.max_seq_len:
            # We take the MOST RECENT transactions.
            group = group.sort_values('date', ascending=True).iloc[-self.config.max_seq_len:]

        # 1. Text Features
        texts = (group['bankRawDescription'].fillna('') + " " + group['counter_party'].fillna('')).tolist()

        # 2. Amount Features
        amounts = group['amount'].values.astype(np.float32)
        # Log normalize
        log_amounts = np.log1p(np.abs(amounts)) * np.sign(amounts)

        # 3. Date Features
        dates = pd.to_datetime(group['date'])
        days_since_start = (dates - dates.min()).dt.days.values.astype(np.float32)

        # 4. Targets
        pattern_ids = group['patternId'].values
        cycles = group['patternCycle'].map(self.config.cycle_map).fillna(0).values.astype(np.int64)

        return {
            "texts": texts,
            "amounts": log_amounts,
            "days": days_since_start,
            "pattern_ids": pattern_ids,
            "cycles": cycles
        }


def collate_fn(batch: list[dict], tokenizer, config: MultiExpConfig):
    """
    Standard dynamic padding collate function.
    """

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
    text_emb_dim = encodings['input_ids'].shape[1]

    # Initialize Tensors
    batched_input_ids = torch.zeros((batch_size, max_len_in_batch, text_emb_dim), dtype=torch.long)
    batched_attention_mask = torch.zeros((batch_size, max_len_in_batch, text_emb_dim), dtype=torch.long)
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

        # Adjacency
        p_ids = item['pattern_ids']
        p_ids_tensor = torch.tensor(p_ids).unsqueeze(1)
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