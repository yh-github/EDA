import logging
import numpy as np
import pandas as pd
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from multi.config import MultiExpConfig, MultiFieldConfig

logger = logging.getLogger(__name__)


def get_dataloader(df: pd.DataFrame, config: MultiExpConfig, shuffle: bool = True, n_workers=4) -> DataLoader:
    """
    Factory function to create a DataLoader from a DataFrame.
    """
    # Use fast tokenizer if available
    # use_fast=True is default, but explicit is good.
    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_model, use_fast=True)

    # Pass a copy to ensure we don't mutate the external DF
    dataset = MultiTransactionDataset(df.copy(), config, tokenizer)

    collate = partial(collate_fn, tokenizer=tokenizer, config=config)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=n_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=(n_workers > 0)
    )


def analyze_token_distribution(df: pd.DataFrame, tokenizer, config: MultiExpConfig):
    """
    Analyzes and logs statistics about token lengths for text and counter_party.
    """
    fields = MultiFieldConfig()

    logger.info("=" * 60)
    logger.info("📊 TOKEN DISTRIBUTION ANALYSIS")
    logger.info("=" * 60)

    def report_stats(name, texts, max_len):
        if not texts:
            logger.info(f"Feature '{name}' is empty or disabled.")
            return

        # Batch encode for speed
        encodings = tokenizer(texts, truncation=False, padding=False)['input_ids']
        lengths_np = np.array([len(x) for x in encodings])

        logger.info(f"--- {name} (Max Limit: {max_len}) ---")
        logger.info(f"  Min: {np.min(lengths_np)}")
        logger.info(f"  Max: {np.max(lengths_np)}")
        logger.info(f"  Avg: {np.mean(lengths_np):.2f}")
        logger.info(f"  Std: {np.std(lengths_np):.2f}")
        logger.info(f"  Median: {np.median(lengths_np)}")

        not_truncated = np.sum(lengths_np <= max_len)
        pct_kept = (not_truncated / len(lengths_np)) * 100
        logger.info(f"  ✅ Not Truncated (<={max_len}): {pct_kept:.2f}%")

        p95 = np.percentile(lengths_np, 95)
        p99 = np.percentile(lengths_np, 99)
        logger.info(f"  95th Percentile: {p95}")
        logger.info(f"  99th Percentile: {p99}")
        print("")

    logger.info("Analyzing 'bankRawDescription'...")
    all_texts = df[fields.text].fillna("").astype(str).tolist()
    report_stats("Text", all_texts, config.max_text_length)

    if config.use_counter_party and fields.counter_party in df.columns:
        logger.info("Analyzing 'counter_party'...")
        all_cps = df[fields.counter_party].fillna("").astype(str).tolist()
        non_empty_cps = [x for x in all_cps if x.strip()]
        if non_empty_cps:
            report_stats("CounterParty (Non-Empty)", non_empty_cps, config.max_cp_length)
        else:
            logger.info("CounterParty column exists but contains no data.")
    logger.info("=" * 60)


class MultiTransactionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: MultiExpConfig, tokenizer: PreTrainedTokenizerBase):
        self.config = config
        self.fields = MultiFieldConfig()

        # 1. Global Preprocessing (Vectorized)
        logger.info(f"Preprocessing {len(df)} rows (Vectorized)...")

        # Ensure Date is datetime
        df[self.fields.date] = pd.to_datetime(df[self.fields.date], errors='coerce')

        # Drop invalid dates if any
        if df[self.fields.date].isnull().any():
            logger.warning("Found NaT dates. Dropping invalid rows.")
            df = df.dropna(subset=[self.fields.date])

        # Preserve Index
        if '_true_index' not in df.columns:
            df['_true_index'] = df.index

        # Clean/Encode Labels
        df = self._encode_metadata(df)

        # 2. Pre-calculate Continuous Features (Numpy speed)
        # Amounts
        amounts = df[self.fields.amount].values.astype(np.float32)
        # 0 amounts are treated as Negative (Outgoing) -1
        direction = np.sign(amounts)
        direction[direction == 0] = -1

        self.all_log_amounts = np.log1p(np.abs(amounts)) * direction

        # Dates (Normalized)
        # Normalize to midnight to remove time-of-day noise (CRITICAL for recurring patterns)
        dates = df[self.fields.date]
        normalized_dates = dates.dt.normalize()
        min_date = normalized_dates.min()

        self.all_days = (normalized_dates - min_date).dt.days.values.astype(np.float32)

        # Calendar Features
        dow = normalized_dates.dt.dayofweek.values.astype(np.float32)
        dom = normalized_dates.dt.day.values.astype(np.float32)
        two_pi = 2 * np.pi

        self.all_calendar = np.stack([
            np.sin(dow * (two_pi / 7)),
            np.cos(dow * (two_pi / 7)),
            np.sin(dom * (two_pi / 31)),
            np.cos(dom * (two_pi / 31))
        ], axis=1).astype(np.float32)

        # IDs
        self.all_pattern_ids = df['patternId_encoded'].values.astype(np.int64)
        self.all_cycles = df[self.fields.patternCycle].map(config.cycle_map).fillna(0).values.astype(np.int64)
        self.all_true_indices = df['_true_index'].values.astype(np.int64)

        # 3. Pre-tokenize (Batch Process to avoid OOM on huge datasets)
        logger.info("Tokenizing text data...")
        self.text_input_ids, self.text_attn_mask = self._batch_tokenize(
            df[self.fields.text].fillna("").astype(str).tolist(),
            tokenizer,
            config.max_text_length
        )

        self.cp_input_ids = None
        self.cp_attn_mask = None
        if config.use_counter_party:
            if self.fields.counter_party in df.columns:
                self.cp_input_ids, self.cp_attn_mask = self._batch_tokenize(
                    df[self.fields.counter_party].fillna("").astype(str).tolist(),
                    tokenizer,
                    config.max_cp_length
                )
            else:
                self.config.use_counter_party = False

        # 4. Partitioning Logic (Amount Sort -> Window -> Date Sort)
        # We calculate the INDICES for every window upfront.
        logger.info("Computing Window partitions...")

        # We need a temporary dataframe that aligns with our numpy arrays (0..N)
        # to perform the grouping logic.
        df['__internal_idx__'] = np.arange(len(df))
        df['__direction__'] = direction

        # Sort by Account -> Direction -> Amount (Primary Partitioning Logic)
        # Mergesort is stable, usually preferred for data pipelines
        sort_cols = [self.fields.accountId, '__direction__', self.fields.amount, self.fields.date]
        df_sorted = df.sort_values(by=sort_cols, kind='mergesort')

        self.window_indices = []

        # Group by Account/Direction using the sorted DF
        # sort=False ensures we keep the Amount-sorted order
        grouped = df_sorted.groupby([self.fields.accountId, '__direction__'], sort=False)

        for _, group in grouped:
            # The indices in 'group' are essentially sorted by Amount
            grp_indices = group['__internal_idx__'].values
            grp_dates = group[self.fields.date].values

            n_items = len(grp_indices)
            for i in range(0, n_items, config.max_seq_len):
                # 1. Get the chunk (Amount sorted)
                chunk_indices = grp_indices[i: i + config.max_seq_len]

                # 2. Re-sort this specific chunk by Date (Requirement for Transformer)
                chunk_dates = grp_dates[i: i + config.max_seq_len]
                date_sort_order = np.argsort(chunk_dates)  # Returns indices into the chunk

                # Apply sort
                final_window_indices = chunk_indices[date_sort_order]
                self.window_indices.append(final_window_indices)

        logger.info(f"Dataset ready. {len(self.window_indices)} windows generated.")

    def _encode_metadata(self, df):
        col = self.fields.patternId
        if col not in df.columns: df[col] = -1
        if self.fields.patternCycle not in df.columns: df[self.fields.patternCycle] = 'None'

        # Fast factorization
        df['patternId_str'] = df[col].astype(str)
        codes, _ = pd.factorize(df['patternId_str'])
        df['patternId_encoded'] = codes

        # Mask noise
        noise_mask = df['patternId_str'].isin(['-1', '-1.0', 'None', 'nan', '<NA>'])
        df.loc[noise_mask, 'patternId_encoded'] = -1
        df = df.drop(columns=['patternId_str'])
        return df

    def _batch_tokenize(self, texts, tokenizer, max_len, batch_size=10000):
        """Tokenize in batches to check memory usage and show progress."""
        all_ids = []
        all_masks = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc = tokenizer(
                batch,
                padding='max_length',
                truncation=True,
                max_length=max_len,
                return_tensors='np'  # Numpy is lighter than PT until collation
            )
            all_ids.append(enc['input_ids'])
            all_masks.append(enc['attention_mask'])

        return np.concatenate(all_ids), np.concatenate(all_masks)

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        # Retrieve pre-calculated indices for this window
        indices = self.window_indices[idx]

        # Simple Numpy Slicing (Fast!)
        return {
            "text_ids": self.text_input_ids[indices],
            "text_mask": self.text_attn_mask[indices],
            "cp_ids": self.cp_input_ids[indices] if self.cp_input_ids is not None else None,
            "cp_mask": self.cp_attn_mask[indices] if self.cp_attn_mask is not None else None,
            "amounts": self.all_log_amounts[indices],
            "days": self.all_days[indices],
            "calendar_features": self.all_calendar[indices],
            "pattern_ids": self.all_pattern_ids[indices],
            "cycles": self.all_cycles[indices],
            "original_index": self.all_true_indices[indices]
        }


def collate_fn(batch: list[dict], tokenizer, config: MultiExpConfig):
    # Determine max length in this batch for dynamic padding
    lengths = [len(x['text_ids']) for x in batch]
    max_len = max(lengths)
    batch_size = len(batch)

    # Helper to allocate and pad
    def collate_tensor(key, shape_suffix, dtype, padding_val=0):
        # Create full tensor filled with padding value
        out = torch.full((batch_size, max_len, *shape_suffix), padding_val, dtype=dtype)
        for i, item in enumerate(batch):
            row_len = lengths[i]
            data = torch.from_numpy(item[key])
            out[i, :row_len] = data
        return out

    input_ids = collate_tensor("text_ids", (config.max_text_length,), torch.long)
    attention_mask = collate_tensor("text_mask", (config.max_text_length,), torch.long)

    # Features
    amounts = collate_tensor("amounts", (1,), torch.float32)
    days = collate_tensor("days", (1,), torch.float32)
    calendar = collate_tensor("calendar_features", (4,), torch.float32)
    cycles = collate_tensor("cycles", (), torch.long)
    pattern_ids = collate_tensor("pattern_ids", (), torch.long, padding_val=-1)
    original_index = collate_tensor("original_index", (), torch.long, padding_val=-1)

    # Dynamic Adjacency Matrix
    adjacency = torch.zeros((batch_size, max_len, max_len), dtype=torch.float32)
    padding_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        l = lengths[i]
        p_ids = torch.from_numpy(item['pattern_ids'])

        # Calculate Adjacency: (N, 1) == (1, N) -> (N, N)
        # Ignore noise (-1) matches
        if torch.any(p_ids != -1):
            p_ids_v = p_ids.unsqueeze(1)
            matches = (p_ids_v == p_ids_v.T) & (p_ids_v != -1)
            adjacency[i, :l, :l] = matches.float()

        padding_mask[i, :l] = True

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "amounts": amounts,
        "days": days,
        "calendar_features": calendar,
        "adjacency_target": adjacency,
        "cycle_target": cycles,
        "pattern_ids": pattern_ids,
        "padding_mask": padding_mask,
        "original_index": original_index
    }

    if config.use_counter_party:
        result["cp_input_ids"] = collate_tensor("cp_ids", (config.max_cp_length,), torch.long)
        result["cp_attention_mask"] = collate_tensor("cp_mask", (config.max_cp_length,), torch.long)

    return result