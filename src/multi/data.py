import logging
import numpy as np
import pandas as pd
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from multi.config import MultiExpConfig, MultiFieldConfig

logger = logging.getLogger(__name__)


def get_dataloader(df: pd.DataFrame, config: MultiExpConfig, shuffle: bool = True, n_workers=4) -> DataLoader:
    """
    Factory function to create a DataLoader from a DataFrame.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_model)
    dataset = MultiTransactionDataset(df, config, tokenizer)

    # Bind tokenizer and config to the collate function
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

        encodings = tokenizer(texts, truncation=False, padding=False)
        lengths = [len(ids) for ids in encodings['input_ids']]
        lengths_np = np.array(lengths)

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
    def __init__(self, df: pd.DataFrame, config: MultiExpConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.fields = MultiFieldConfig()

        # Work on a copy to avoid SettingWithCopy warnings on the input df
        df = df.copy()

        # OPTIMIZATION: Convert date to datetime ONCE here.
        # This prevents running pd.to_datetime() repeatedly in __getitem__
        df[self.fields.date] = pd.to_datetime(df[self.fields.date])

        # 0. Preserve True Original Index
        # We do this BEFORE any sorting or resetting so we can always map back to the source CSV rows.
        if '_true_index' not in df.columns:
            df['_true_index'] = df.index

        # 1. Clean / Encode Labels
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

        # 0 amounts are treated as Negative (Outgoing) -1
        df['direction'] = np.sign(df[self.fields.amount])

        zeros_mask = (df['direction'] == 0)
        if zeros_mask.any():
            logger.warning(
                f"Dataset contains {zeros_mask.sum()} transactions with amount=0. "
                "Treating them as negative direction (-1) instead of filtering."
            )
            df.loc[zeros_mask, 'direction'] = -1

        # --- STRATEGY: Amount-Sorted Partitioning ---
        # We group similar amounts together.
        # Global Sort: Account -> Direction -> Amount -> Date
        logger.info("Sorting dataset by [Account, Direction, Amount, Date] for partitioning...")
        df = df.sort_values(
            by=[self.fields.accountId, 'direction', self.fields.amount, self.fields.date],
            ascending=[True, True, True, True]
        )

        # 2. Reset Index to align with Tensor indexing (0 to N-1)
        # This 'internal_index' matches the order of pre-tokenized tensors.
        df = df.reset_index(drop=True)

        # 3. Pre-tokenize
        # Tokenization happens on the sorted dataframe, so index 0 here matches row 0 in df.
        logger.info("Pre-tokenizing Description...")
        all_texts = df[self.fields.text].fillna("").astype(str).tolist()
        self.text_enc = tokenizer(
            all_texts,
            padding='max_length',
            truncation=True,
            max_length=config.max_text_length,
            return_tensors='pt'
        )

        self.cp_enc = None
        if self.config.use_counter_party:
            if self.fields.counter_party in df.columns:
                logger.info("Pre-tokenizing CounterParty...")
                all_cps = df[self.fields.counter_party].fillna("").astype(str).tolist()
                self.cp_enc = tokenizer(
                    all_cps,
                    padding='max_length',
                    truncation=True,
                    max_length=config.max_cp_length,
                    return_tensors='pt'
                )
            else:
                logger.warning("CounterParty enabled but column missing. Creating empty placeholders.")
                self.config.use_counter_party = False

        # 4. Window Generation (Partitioning)
        # We slice the sorted dataframe into NON-OVERLAPPING windows.
        self.samples = []

        # Group by Account/Direction (preserves the Amount sort order within groups)
        grouped = df.groupby([self.fields.accountId, 'direction'], sort=False)

        for _, group in grouped:
            n_items = len(group)
            # Slice into chunks of max_seq_len
            for i in range(0, n_items, self.config.max_seq_len):
                # We store the dataframe slice.
                # .copy() ensures we can modify it (sort it) without affecting the original if needed
                window = group.iloc[i: i + self.config.max_seq_len].copy()
                self.samples.append(window)

        logger.info(f"Partitioned dataset into {len(self.samples)} windows. (100% data coverage)")

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
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. Retrieve the window (Currently sorted by Amount)
        window = self.samples[idx]
        f = self.fields

        # 2. Local Re-Sort by Date (CRITICAL)
        # The Transformer needs chronological order to learn periodicity.
        # We re-sort the slice by Date before feeding it to the model.
        window = window.sort_values(f.date, ascending=True)

        # Indices in the internal dataset (0..N) used for looking up pre-tokenized tensors
        internal_indices = window.index.values

        # --- KEY: Retrieve the REAL original indices for evaluation ---
        true_indices = window['_true_index'].values

        # Text Tensors (lookup using internal index)
        text_input_ids = self.text_enc['input_ids'][internal_indices]
        text_attn_mask = self.text_enc['attention_mask'][internal_indices]

        cp_input_ids = None
        cp_attn_mask = None
        if self.config.use_counter_party and self.cp_enc is not None:
            cp_input_ids = self.cp_enc['input_ids'][internal_indices]
            cp_attn_mask = self.cp_enc['attention_mask'][internal_indices]

        # Numeric Features
        amounts = window[f.amount].values.astype(np.float32)
        log_amounts = np.log1p(np.abs(amounts)) * np.sign(amounts)

        # --- DATE HANDLING (Normalized) ---
        # 1. 'raw_dates' are already datetime objects (converted in __init__)
        raw_dates = window[f.date]

        # 2. Normalize to midnight to remove time-of-day jitter
        #    This makes 'days_since_start' represent calendar days elapsed.
        normalized_dates = raw_dates.dt.normalize()

        min_date = normalized_dates.iloc[0]
        days_since_start = (normalized_dates - min_date).dt.days.values.astype(np.float32)

        # Day of Week / Month calculations work fine on normalized dates
        dow = normalized_dates.dt.dayofweek.values.astype(np.float32)
        dom = normalized_dates.dt.day.values.astype(np.float32)

        two_pi = 2 * np.pi
        calendar_feats = np.stack([
            np.sin(dow * (two_pi / 7)),
            np.cos(dow * (two_pi / 7)),
            np.sin(dom * (two_pi / 31)),
            np.cos(dom * (two_pi / 31))
        ], axis=1).astype(np.float32)

        pattern_ids = window['patternId_encoded'].values.astype(np.int64)
        cycles = window[f.patternCycle].map(self.config.cycle_map).fillna(0).values.astype(np.int64)

        return {
            "text_ids": text_input_ids,
            "text_mask": text_attn_mask,
            "cp_ids": cp_input_ids,
            "cp_mask": cp_attn_mask,
            "amounts": log_amounts,
            "days": days_since_start,
            "calendar_features": calendar_feats,
            "pattern_ids": pattern_ids,
            "cycles": cycles,
            # We return the TRUE dataframe indices now for robust evaluation
            "original_index": torch.tensor(true_indices, dtype=torch.long)
        }


def collate_fn(batch: list[dict], tokenizer, config: MultiExpConfig):
    is_pre_tokenized = 'text_ids' in batch[0]
    batch_size = len(batch)

    if is_pre_tokenized:
        account_lengths = [len(item['text_ids']) for item in batch]
    else:
        account_lengths = [len(item['texts']) for item in batch]

    max_len_in_batch = max(account_lengths)

    # 1. Text
    seq_len_text = config.max_text_length
    b_input_ids = torch.zeros((batch_size, max_len_in_batch, seq_len_text), dtype=torch.long)
    b_attn_mask = torch.zeros((batch_size, max_len_in_batch, seq_len_text), dtype=torch.long)

    # 2. CP
    b_cp_ids = None
    b_cp_mask = None
    seq_len_cp = config.max_cp_length

    if config.use_counter_party:
        b_cp_ids = torch.zeros((batch_size, max_len_in_batch, seq_len_cp), dtype=torch.long)
        b_cp_mask = torch.zeros((batch_size, max_len_in_batch, seq_len_cp), dtype=torch.long)

    # 3. Other Tensors
    b_amounts = torch.zeros((batch_size, max_len_in_batch, 1), dtype=torch.float32)
    b_days = torch.zeros((batch_size, max_len_in_batch, 1), dtype=torch.float32)
    b_calendar = torch.zeros((batch_size, max_len_in_batch, 4), dtype=torch.float32)
    b_cycles = torch.zeros((batch_size, max_len_in_batch), dtype=torch.long)
    b_adjacency = torch.zeros((batch_size, max_len_in_batch, max_len_in_batch), dtype=torch.float32)
    b_pattern_ids = torch.full((batch_size, max_len_in_batch), -1, dtype=torch.long)
    padding_mask = torch.zeros((batch_size, max_len_in_batch), dtype=torch.bool)

    # Init with -1
    b_original_index = torch.full((batch_size, max_len_in_batch), -1, dtype=torch.long)

    if not is_pre_tokenized:
        all_texts = [t for item in batch for t in item['texts']]
        enc_text = tokenizer(all_texts, padding='max_length', truncation=True, max_length=config.max_text_length,
                             return_tensors="pt")
        if config.use_counter_party:
            all_cps = [t for item in batch for t in item['cps']]
            enc_cp = tokenizer(all_cps, padding='max_length', truncation=True, max_length=config.max_cp_length,
                               return_tensors="pt")

    current_idx_inf = 0

    for i, (item, length) in enumerate(zip(batch, account_lengths)):
        if is_pre_tokenized:
            b_input_ids[i, :length] = item['text_ids']
            b_attn_mask[i, :length] = item['text_mask']
            if config.use_counter_party and item['cp_ids'] is not None:
                b_cp_ids[i, :length] = item['cp_ids']
                b_cp_mask[i, :length] = item['cp_mask']
        else:
            b_input_ids[i, :length] = enc_text['input_ids'][current_idx_inf: current_idx_inf + length]
            b_attn_mask[i, :length] = enc_text['attention_mask'][current_idx_inf: current_idx_inf + length]
            if config.use_counter_party:
                b_cp_ids[i, :length] = enc_cp['input_ids'][current_idx_inf: current_idx_inf + length]
                b_cp_mask[i, :length] = enc_cp['attention_mask'][current_idx_inf: current_idx_inf + length]
            current_idx_inf += length

        b_amounts[i, :length, 0] = torch.tensor(item['amounts'])
        b_days[i, :length, 0] = torch.tensor(item['days'])
        b_calendar[i, :length, :] = torch.tensor(item['calendar_features'])
        b_cycles[i, :length] = torch.tensor(item['cycles'])

        # --- Include original_index in batch ---
        b_original_index[i, :length] = item['original_index']

        p_ids = item['pattern_ids']
        b_pattern_ids[i, :length] = torch.tensor(p_ids)

        if np.any(p_ids != -1):
            p_ids_tensor = torch.tensor(p_ids).unsqueeze(1)
            matches = (p_ids_tensor == p_ids_tensor.T) & (p_ids_tensor != -1)
            b_adjacency[i, :length, :length] = matches.float()

        padding_mask[i, :length] = True

    result = {
        "input_ids": b_input_ids,
        "attention_mask": b_attn_mask,
        "amounts": b_amounts,
        "days": b_days,
        "calendar_features": b_calendar,
        "adjacency_target": b_adjacency,
        "cycle_target": b_cycles,
        "pattern_ids": b_pattern_ids,
        "padding_mask": padding_mask,
        "original_index": b_original_index
    }

    if config.use_counter_party:
        result["cp_input_ids"] = b_cp_ids
        result["cp_attention_mask"] = b_cp_mask

    return result