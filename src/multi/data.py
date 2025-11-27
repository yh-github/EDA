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
        self.fields = MultiFieldConfig()

        if self.fields.patternCycle not in df.columns:
            df[self.fields.patternCycle] = 'None'
        if self.fields.patternId not in df.columns:
            df[self.fields.patternId] = -1

        df[self.fields.patternCycle] = df[self.fields.patternCycle].fillna('None')

        valid_cycles = set(config.cycle_map.keys())
        mask = df[self.fields.patternCycle].isin(valid_cycles)
        if not mask.all():
            df = df[mask].copy()

        df = self._encode_pattern_ids(df)

        df['direction'] = np.sign(df[self.fields.amount])
        df = df[df['direction'] != 0]

        self.groups = [group for _, group in df.groupby([self.fields.accountId, 'direction'])]

    def _encode_pattern_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        col = self.fields.patternId
        df['patternId_str'] = df[col].astype(str)
        codes, _ = pd.factorize(df['patternId_str'])
        df['patternId_encoded'] = codes

        noise_mask = df['patternId_str'].isin(['-1', '-1.0', 'None', 'nan', '<NA>'])
        df.loc[noise_mask, 'patternId_encoded'] = -1
        df = df.drop(columns=['patternId_str'])
        return df

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        f = self.fields

        if len(group) > self.config.max_seq_len:
            group = group.sort_values(f.date, ascending=True).iloc[-self.config.max_seq_len:]
        else:
            group = group.sort_values(f.date, ascending=True)

        texts = group[f.text].fillna('').tolist()

        cps = []
        if self.config.use_counter_party:
            if f.counter_party in group.columns:
                cps = group[f.counter_party].fillna('').tolist()
            else:
                cps = [""] * len(texts)

        amounts = group[f.amount].values.astype(np.float32)
        log_amounts = np.log1p(np.abs(amounts)) * np.sign(amounts)

        dates = pd.to_datetime(group[f.date])
        min_date = dates.iloc[0]
        # 1. Existing Relative Time (For Frequency/Intervals)
        days_since_start = (dates - min_date).dt.days.values.astype(np.float32)

        # 2. NEW: Calendar Features (For Phase/Seasonality)
        # Day of Week (0=Mon, 6=Sun)
        dow = dates.dt.dayofweek.values.astype(np.float32)
        # Day of Month (1..31)
        dom = dates.dt.day.values.astype(np.float32)

        # Normalize to [0, 2π]
        # We use 7 days for week, 31 days for month to ensure coverage
        two_pi = 2 * np.pi

        # 4 Dimensions: Sin/Cos Week, Sin/Cos Month
        # Shape: [Seq_Len, 4]
        calendar_feats = np.stack([
            np.sin(dow * (two_pi / 7)),
            np.cos(dow * (two_pi / 7)),
            np.sin(dom * (two_pi / 31)),
            np.cos(dom * (two_pi / 31))
        ], axis=1).astype(np.float32)

        pattern_ids = group['patternId_encoded'].values.astype(np.int64)
        cycles = group[f.patternCycle].map(self.config.cycle_map).fillna(0).values.astype(np.int64)

        return {
            "texts": texts,
            "cps": cps,
            "amounts": log_amounts,
            "days": days_since_start,
            "calendar_features": calendar_feats,
            "pattern_ids": pattern_ids,
            "cycles": cycles
        }


def collate_fn(batch: list[dict], tokenizer, config: MultiExpConfig):
    all_texts = [t for item in batch for t in item['texts']]
    account_lengths = [len(item['texts']) for item in batch]
    max_len_in_batch = max(account_lengths)
    batch_size = len(batch)

    # 1. Main Text Tokenization
    enc_text = tokenizer(
        all_texts,
        padding=True,
        truncation=True,
        max_length=config.max_text_length,
        return_tensors="pt"
    )

    # 2. Counter Party Tokenization (Conditional)
    enc_cp = None
    if config.use_counter_party:
        all_cps = [t for item in batch for t in item['cps']]
        enc_cp = tokenizer(
            all_cps,
            padding=True,
            truncation=True,
            max_length=config.max_cp_length,
            return_tensors="pt"
        )

    # Initialize Tensors
    seq_len_text = enc_text['input_ids'].shape[1]
    b_input_ids = torch.zeros((batch_size, max_len_in_batch, seq_len_text), dtype=torch.long)
    b_attn_mask = torch.zeros((batch_size, max_len_in_batch, seq_len_text), dtype=torch.long)

    b_cp_ids = None
    b_cp_mask = None

    if config.use_counter_party and enc_cp is not None:
        seq_len_cp = enc_cp['input_ids'].shape[1]
        b_cp_ids = torch.zeros((batch_size, max_len_in_batch, seq_len_cp), dtype=torch.long)
        b_cp_mask = torch.zeros((batch_size, max_len_in_batch, seq_len_cp), dtype=torch.long)

    b_amounts = torch.zeros((batch_size, max_len_in_batch, 1), dtype=torch.float32)
    b_days = torch.zeros((batch_size, max_len_in_batch, 1), dtype=torch.float32)
    # New: Calendar Buffer [Batch, Max_Len, 4]
    b_calendar = torch.zeros((batch_size, max_len_in_batch, 4), dtype=torch.float32)

    b_cycles = torch.zeros((batch_size, max_len_in_batch), dtype=torch.long)
    b_adjacency = torch.zeros((batch_size, max_len_in_batch, max_len_in_batch), dtype=torch.float32)
    padding_mask = torch.zeros((batch_size, max_len_in_batch), dtype=torch.bool)

    current_idx = 0
    for i, (item, length) in enumerate(zip(batch, account_lengths)):
        # Text
        b_input_ids[i, :length] = enc_text['input_ids'][current_idx: current_idx + length]
        b_attn_mask[i, :length] = enc_text['attention_mask'][current_idx: current_idx + length]

        # CP
        if config.use_counter_party and b_cp_ids is not None:
            b_cp_ids[i, :length] = enc_cp['input_ids'][current_idx: current_idx + length]
            b_cp_mask[i, :length] = enc_cp['attention_mask'][current_idx: current_idx + length]

        # Scalars
        b_amounts[i, :length, 0] = torch.tensor(item['amounts'])
        b_days[i, :length, 0] = torch.tensor(item['days'])
        # Calendar
        b_calendar[i, :length, :] = torch.tensor(item['calendar_features'])

        b_cycles[i, :length] = torch.tensor(item['cycles'])

        # Adjacency
        p_ids = item['pattern_ids']
        if np.any(p_ids != -1):
            p_ids_tensor = torch.tensor(p_ids).unsqueeze(1)
            matches = (p_ids_tensor == p_ids_tensor.T) & (p_ids_tensor != -1)
            b_adjacency[i, :length, :length] = matches.float()

        padding_mask[i, :length] = True
        current_idx += length

    result = {
        "input_ids": b_input_ids,
        "attention_mask": b_attn_mask,
        "amounts": b_amounts,
        "days": b_days,
        "calendar_features": b_calendar,
        "adjacency_target": b_adjacency,
        "cycle_target": b_cycles,
        "padding_mask": padding_mask
    }

    if config.use_counter_party:
        result["cp_input_ids"] = b_cp_ids
        result["cp_attention_mask"] = b_cp_mask

    return result


def get_dataloader(df, config: MultiExpConfig, shuffle=True):
    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_model)
    ds = MultiTransactionDataset(df, config, tokenizer)
    return DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: collate_fn(x, tokenizer, config)
    )