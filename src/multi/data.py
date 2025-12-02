import logging
from functools import partial
from typing import Optional, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from multi.config import MultiExpConfig, MultiFieldConfig

logger = logging.getLogger(__name__)


class TransactionSample(TypedDict):
    """Output of Dataset.__getitem__ (Numpy Arrays)"""
    text_ids: npt.NDArray[np.int64]
    text_mask: npt.NDArray[np.int64]
    cp_ids: Optional[npt.NDArray[np.int64]]
    cp_mask: Optional[npt.NDArray[np.int64]]
    amounts: npt.NDArray[np.float32]
    days: npt.NDArray[np.float32]
    calendar_features: npt.NDArray[np.float32]
    pattern_ids: npt.NDArray[np.int64]
    cycles: npt.NDArray[np.int64]
    original_index: npt.NDArray[np.int64]


class TransactionBatch(TypedDict):
    """Output of collate_fn (PyTorch Tensors)"""
    input_ids: torch.Tensor  # [B, S, L_text]
    attention_mask: torch.Tensor  # [B, S, L_text]
    cp_input_ids: Optional[torch.Tensor]
    cp_attention_mask: Optional[torch.Tensor]
    amounts: torch.Tensor  # [B, S, 1]
    days: torch.Tensor  # [B, S, 1]
    calendar_features: torch.Tensor  # [B, S, 6]
    adjacency_target: torch.Tensor  # [B, S, S]
    cycle_target: torch.Tensor  # [B, S]
    pattern_ids: torch.Tensor  # [B, S]
    padding_mask: torch.Tensor  # [B, S] (Bool)
    original_index: torch.Tensor  # [B, S]


# ------------------------

def get_dataloader(
        df: pd.DataFrame,
        config: MultiExpConfig,
        shuffle: bool = True,
        n_workers: int = 4,
        oversample_positives: bool = True
) -> DataLoader:
    """
    Factory function to create a DataLoader.

    If 'oversample_positives' is True and shuffle is True (Training),
    it uses a WeightedRandomSampler to ensure batches contain meaningful data.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_model, use_fast=True)
    dataset = MultiTransactionDataset(df.copy(), config, tokenizer)
    collate = partial(collate_fn, config=config)

    sampler = None
    if shuffle and oversample_positives:
        # Calculate weights for Stratified Sampling
        # We want to boost windows that have at least one recurring transaction

        has_positives = dataset.window_has_positive_signal.astype(float)
        num_pos = np.sum(has_positives)
        num_neg = len(has_positives) - num_pos

        if num_pos > 0:
            # Standard Inverse Frequency Weighting
            weight_pos = 1.0 / num_pos
            weight_neg = 1.0 / num_neg

            # Boost positives significantly (e.g., aim for 50/50 mix in batch)
            weights = np.where(has_positives > 0, weight_pos, weight_neg)

            logger.info(f"Applying WeightedRandomSampler: {num_pos} Positive Windows vs {num_neg} Negative Windows")

            # Create Sampler
            # replacement=True is required for WeightedRandomSampler
            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(weights),
                num_samples=len(weights),
                replacement=True
            )
            # Shuffle must be False if sampler is used
            shuffle = False

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,  # False if sampler is active
        sampler=sampler,
        collate_fn=collate,
        num_workers=n_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=(n_workers > 0)
    )


class MultiTransactionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: MultiExpConfig, tokenizer: PreTrainedTokenizerBase):
        self.config = config
        self.fields = MultiFieldConfig()

        logger.info(f"Preprocessing {len(df)} rows (Vectorized)...")

        # Ensure Date
        df[self.fields.date] = pd.to_datetime(df[self.fields.date], errors='coerce')
        if df[self.fields.date].isnull().any():
            logger.warning("Found NaT dates. Dropping invalid rows.")
            df = df.dropna(subset=[self.fields.date])

        if '_true_index' not in df.columns:
            df['_true_index'] = df.index

        df = self._encode_metadata(df)

        # --- SORTING (Source of Truth) ---
        temp_amounts = df[self.fields.amount].values.astype(np.float32)
        direction = np.sign(temp_amounts)
        direction[direction == 0] = -1
        df['__direction__'] = direction

        logger.info("Sorting dataset by [Account, Direction, Amount, Date]...")
        sort_cols = [self.fields.accountId, '__direction__', self.fields.amount, self.fields.date]
        df = df.sort_values(by=sort_cols, kind='mergesort').reset_index(drop=True)

        # 2. Features
        amounts = df[self.fields.amount].values.astype(np.float32)
        direction = np.sign(amounts)
        direction[direction == 0] = -1

        self.all_log_amounts: npt.NDArray[np.float32] = np.log1p(np.abs(amounts)) * direction

        dates = df[self.fields.date]
        normalized_dates = dates.dt.normalize()
        min_date = normalized_dates.min()
        self.all_days: npt.NDArray[np.float32] = (normalized_dates - min_date).dt.days.values.astype(np.float32)

        dow = normalized_dates.dt.dayofweek.values.astype(np.float32)
        dom = normalized_dates.dt.day.values.astype(np.float32)

        # We use a fixed epoch to ensure 'Week A' vs 'Week B' is consistent globally
        epoch_date = pd.Timestamp("2000-01-01")
        # Days since epoch
        # noinspection PyTypeChecker
        global_days = (normalized_dates - epoch_date).dt.days.values
        cycle_14 = (global_days % 14).astype(np.float32)

        two_pi = 2 * np.pi

        self.all_calendar: npt.NDArray[np.float32] = np.stack([
            np.sin(dow * (two_pi / 7)),
            np.cos(dow * (two_pi / 7)),
            np.sin(dom * (two_pi / 31)),
            np.cos(dom * (two_pi / 31)),
            np.sin(cycle_14 * (two_pi / 14)),
            np.cos(cycle_14 * (two_pi / 14))
        ], axis=1).astype(np.float32)

        self.all_pattern_ids: npt.NDArray[np.int64] = df['patternId_encoded'].values.astype(np.int64)
        self.all_cycles: npt.NDArray[np.int64] = df[self.fields.patternCycle].map(config.cycle_map).fillna(
            0).values.astype(np.int64)
        self.all_true_indices: npt.NDArray[np.int64] = df['_true_index'].values.astype(np.int64)

        # 3. Tokenize
        logger.info("Tokenizing text data...")
        self.text_input_ids, self.text_attn_mask = self._batch_tokenize(
            df[self.fields.text].fillna("").astype(str).tolist(),
            tokenizer,
            config.max_text_length
        )

        self.cp_input_ids: Optional[npt.NDArray[np.int64]] = None
        self.cp_attn_mask: Optional[npt.NDArray[np.int64]] = None

        if config.use_counter_party:
            if self.fields.counter_party in df.columns:
                self.cp_input_ids, self.cp_attn_mask = self._batch_tokenize(
                    df[self.fields.counter_party].fillna("").astype(str).tolist(),
                    tokenizer,
                    config.max_cp_length
                )
            else:
                self.config.use_counter_party = False

        # 4. Partitioning
        logger.info("Computing Window partitions...")
        df['__internal_idx__'] = np.arange(len(df))

        self.window_indices: list[npt.NDArray[np.int64]] = []

        # We also need to track which windows have ANY signal for the Sampler
        self.window_has_positive_signal = []

        grouped = df.groupby([self.fields.accountId, '__direction__'], sort=False)

        for _, group in grouped:
            grp_indices = group['__internal_idx__'].values
            grp_days = self.all_days[grp_indices]

            # Check cycles for this group to identify positive windows
            grp_cycles = self.all_cycles[grp_indices]

            n_items = len(grp_indices)
            for i in range(0, n_items, config.max_seq_len):
                chunk_indices = grp_indices[i: i + config.max_seq_len]
                chunk_days = grp_days[i: i + config.max_seq_len]

                # Check signal BEFORE sorting (order doesn't matter for existence)
                chunk_cycles = grp_cycles[i: i + config.max_seq_len]
                has_signal = np.any(chunk_cycles > 0)
                self.window_has_positive_signal.append(has_signal)

                # Re-sort by Date
                date_sort_order = np.argsort(chunk_days)
                final_window_indices = chunk_indices[date_sort_order]
                self.window_indices.append(final_window_indices)

        self.window_has_positive_signal = np.array(self.window_has_positive_signal, dtype=bool)
        logger.info(f"Dataset ready. {len(self.window_indices)} windows generated.")
        logger.info(f"Windows with Signal: {np.sum(self.window_has_positive_signal)} / {len(self.window_indices)}")

    def _encode_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        col = self.fields.patternId
        if col not in df.columns: df[col] = -1
        if self.fields.patternCycle not in df.columns: df[self.fields.patternCycle] = 'None'

        df['patternId_str'] = df[col].astype(str)
        codes, _ = pd.factorize(df['patternId_str'])
        df['patternId_encoded'] = codes

        noise_mask = df['patternId_str'].isin(['-1', '-1.0', 'None', 'nan', '<NA>'])
        df.loc[noise_mask, 'patternId_encoded'] = -1
        df = df.drop(columns=['patternId_str'])
        return df

    @staticmethod
    def _batch_tokenize(texts: list[str], tokenizer: PreTrainedTokenizerBase, max_len: int, batch_size: int = 10000) -> \
            tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        all_ids = []
        all_masks = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc = tokenizer(
                batch, padding='max_length', truncation=True, max_length=max_len, return_tensors='np'
            )
            all_ids.append(enc['input_ids'])
            all_masks.append(enc['attention_mask'])
        if len(all_ids) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        return np.concatenate(all_ids), np.concatenate(all_masks)

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, idx: int) -> TransactionSample:
        indices = self.window_indices[idx]

        # Calculate Window-Relative Days
        raw_days = self.all_days[indices]
        # Robustly handle empty windows (though logic should prevent them)
        if len(raw_days) > 0:
            days_relative = (raw_days - raw_days.min()).astype(np.float32)
        else:
            days_relative = np.array([], dtype=np.float32)

        sample: TransactionSample = {
            "text_ids": self.text_input_ids[indices],
            "text_mask": self.text_attn_mask[indices],
            "cp_ids": self.cp_input_ids[indices] if self.cp_input_ids is not None else None,
            "cp_mask": self.cp_attn_mask[indices] if self.cp_attn_mask is not None else None,
            "amounts": self.all_log_amounts[indices],
            "days": days_relative,
            "calendar_features": self.all_calendar[indices],
            "pattern_ids": self.all_pattern_ids[indices],
            "cycles": self.all_cycles[indices],
            "original_index": self.all_true_indices[indices]
        }
        return sample


def collate_fn(batch: list[TransactionSample], config: MultiExpConfig) -> TransactionBatch:
    lengths = [len(x['text_ids']) for x in batch]
    max_len = max(lengths)
    batch_size = len(batch)

    def collate_tensor(key: str, shape_suffix: tuple[int, ...], dtype: torch.dtype,
                       padding_val: int | float = 0) -> torch.Tensor:
        out = torch.full((batch_size, max_len, *shape_suffix), padding_val, dtype=dtype)
        for _i, _item in enumerate(batch):
            row_len = lengths[_i]
            data_np = _item[key]  # type: ignore
            if data_np is None: continue
            data = torch.from_numpy(data_np)
            if shape_suffix and data.ndim == 1:
                data = data.view(-1, *shape_suffix)
            out[_i, :row_len] = data
        return out

    input_ids = collate_tensor("text_ids", (config.max_text_length,), torch.long)
    attention_mask = collate_tensor("text_mask", (config.max_text_length,), torch.long)
    amounts = collate_tensor("amounts", (1,), torch.float32)
    days = collate_tensor("days", (1,), torch.float32)
    calendar = collate_tensor("calendar_features", (6,), torch.float32)
    cycles = collate_tensor("cycles", (), torch.long)
    pattern_ids = collate_tensor("pattern_ids", (), torch.long, padding_val=-1)
    original_index = collate_tensor("original_index", (), torch.long, padding_val=-1)

    adjacency = torch.zeros((batch_size, max_len, max_len), dtype=torch.float32)
    padding_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        l = lengths[i]
        p_ids = torch.from_numpy(item['pattern_ids'])
        if torch.any(p_ids != -1):
            p_ids_v = p_ids.unsqueeze(1)
            matches = (p_ids_v == p_ids_v.T) & (p_ids_v != -1)
            adjacency[i, :l, :l] = matches.float()
        padding_mask[i, :l] = True

    result: TransactionBatch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "amounts": amounts,
        "days": days,
        "calendar_features": calendar,
        "adjacency_target": adjacency,
        "cycle_target": cycles,
        "pattern_ids": pattern_ids,
        "padding_mask": padding_mask,
        "original_index": original_index,
        "cp_input_ids": None,
        "cp_attention_mask": None
    }

    if config.use_counter_party:
        result["cp_input_ids"] = collate_tensor("cp_ids", (config.max_cp_length,), torch.long)
        result["cp_attention_mask"] = collate_tensor("cp_mask", (config.max_cp_length,), torch.long)

    return result


def analyze_token_distribution(df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, config: MultiExpConfig):
    fields = MultiFieldConfig()
    logger.info("=" * 60)
    logger.info("📊 TOKEN DISTRIBUTION ANALYSIS")

    def report_stats(name: str, texts: list[str], max_len: int):
        if not texts:
            logger.info(f"Feature '{name}' is empty or disabled.")
            return
        encodings = tokenizer(texts, truncation=False, padding=False)['input_ids']
        lengths_np = np.array([len(x) for x in encodings])
        logger.info(f"--- {name} (Max Limit: {max_len}) ---")
        logger.info(f"  Avg: {np.mean(lengths_np):.2f} | P99: {np.percentile(lengths_np, 99)}")
        not_truncated = np.sum(lengths_np <= max_len)
        logger.info(f"  ✅ Kept: {(not_truncated / len(lengths_np)) * 100:.2f}%")

    logger.info("Analyzing 'bankRawDescription'...")
    all_texts = df[fields.text].fillna("").astype(str).tolist()
    report_stats("Text", all_texts, config.max_text_length)
    if config.use_counter_party and fields.counter_party in df.columns:
        logger.info("Analyzing 'counter_party'...")
        all_cps = df[fields.counter_party].fillna("").astype(str).tolist()
        non_empty_cps = [x for x in all_cps if x.strip()]
        if non_empty_cps:
            report_stats("CounterParty (Non-Empty)", non_empty_cps, config.max_cp_length)
    logger.info("=" * 60)