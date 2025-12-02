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
from common.embedder import EmbeddingService

logger = logging.getLogger(__name__)


class TransactionSample(TypedDict):
    """Output of Dataset.__getitem__ (Numpy Arrays)"""
    # Standard Tokenizer Path
    text_ids: Optional[npt.NDArray[np.int64]]
    text_mask: Optional[npt.NDArray[np.int64]]
    cp_ids: Optional[npt.NDArray[np.int64]]
    cp_mask: Optional[npt.NDArray[np.int64]]

    # Cached Embedding Path
    cached_text: Optional[npt.NDArray[np.float32]]
    cached_cp: Optional[npt.NDArray[np.float32]]

    # Shared
    amounts: npt.NDArray[np.float32]
    days: npt.NDArray[np.float32]
    calendar_features: npt.NDArray[np.float32]
    pattern_ids: npt.NDArray[np.int64]
    cycles: npt.NDArray[np.int64]
    original_index: npt.NDArray[np.int64]


class TransactionBatch(TypedDict):
    """Output of collate_fn (PyTorch Tensors)"""
    # Optional inputs based on mode
    input_ids: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor]
    cp_input_ids: Optional[torch.Tensor]
    cp_attention_mask: Optional[torch.Tensor]
    cached_text: Optional[torch.Tensor]
    cached_cp: Optional[torch.Tensor]

    # Shared
    amounts: torch.Tensor
    days: torch.Tensor
    calendar_features: torch.Tensor
    adjacency_target: Optional[torch.Tensor]
    cycle_target: torch.Tensor
    pattern_ids: torch.Tensor
    padding_mask: torch.Tensor
    original_index: torch.Tensor


# --- CACHES ---
_TOKENIZER_CACHE = {}


def get_tokenizer_cached(model_name: str):
    if model_name not in _TOKENIZER_CACHE:
        logger.info(f"Loading Tokenizer: {model_name} (First time)")
        _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return _TOKENIZER_CACHE[model_name]


def get_dataloader(
        df: pd.DataFrame,
        config: MultiExpConfig,
        shuffle: bool = True,
        n_workers: int = 4,
        oversample_positives: bool = True,
        # Option to reuse an existing dataset
        dataset: 'MultiTransactionDataset' = None,
        # Option to pass embedding service for cached mode
        embedding_service: Optional[EmbeddingService] = None
) -> DataLoader:
    """
    Factory function to create a DataLoader.
    Supports both Tokenizer (standard) and EmbeddingService (cached) paths.
    """
    if dataset is None:
        if config.use_cached_embeddings:
            if embedding_service is None:
                # If not provided, create a temp one (warn: might re-load model)
                logger.warning("use_cached_embeddings=True but no EmbeddingService provided. Initializing new one.")
                params = EmbeddingService.Params(model_name=config.emb_model)
                embedding_service = EmbeddingService.create(params)
            dataset = MultiTransactionDataset(df.copy(), config, embedding_service=embedding_service)
        else:
            tokenizer = get_tokenizer_cached(config.text_encoder_model)
            dataset = MultiTransactionDataset(df.copy(), config, tokenizer=tokenizer)

    collate = partial(collate_fn, config=config)

    sampler = None
    if shuffle and oversample_positives:
        # Calculate weights for Stratified Sampling
        has_positives = dataset.window_has_positive_signal.astype(float)
        num_pos = np.sum(has_positives)
        num_neg = len(has_positives) - num_pos

        if num_pos > 0:
            weight_pos = 1.0 / num_pos
            weight_neg = 1.0 / num_neg
            weights = np.where(has_positives > 0, weight_pos, weight_neg)

            logger.info(f"Applying WeightedRandomSampler: {num_pos} Positive Windows vs {num_neg} Negative Windows")

            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(weights),
                num_samples=len(weights),
                replacement=True
            )
            shuffle = False

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate,
        num_workers=n_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=(n_workers > 0)
    )


class MultiTransactionDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 config: MultiExpConfig,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 embedding_service: Optional[EmbeddingService] = None):

        self.config = config
        self.fields = MultiFieldConfig()

        # Validation
        if config.use_cached_embeddings and embedding_service is None:
            raise ValueError("Config expects cached embeddings but no EmbeddingService provided.")
        if not config.use_cached_embeddings and tokenizer is None:
            raise ValueError("Config expects standard tokenization but no Tokenizer provided.")

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

        epoch_date = pd.Timestamp("2000-01-01")
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

        # 3. Text Handling (Tokenize OR Cache)
        text_list = df[self.fields.text].fillna("").astype(str).tolist()

        self.text_input_ids = None
        self.text_attn_mask = None
        self.cached_text = None

        if config.use_cached_embeddings:
            logger.info("Using EmbeddingService to pre-compute/fetch Text Embeddings...")
            self.cached_text = embedding_service.embed(text_list)
        else:
            logger.info("Tokenizing text data...")
            self.text_input_ids, self.text_attn_mask = self._batch_tokenize(
                text_list, tokenizer, config.max_text_length
            )

        # 4. Counter Party Handling
        self.cp_input_ids = None
        self.cp_attn_mask = None
        self.cached_cp = None

        if config.use_counter_party:
            if self.fields.counter_party in df.columns:
                cp_list = df[self.fields.counter_party].fillna("").astype(str).tolist()

                if config.use_cached_embeddings:
                    logger.info("Using EmbeddingService to pre-compute/fetch CounterParty Embeddings...")
                    self.cached_cp = embedding_service.embed(cp_list)
                else:
                    self.cp_input_ids, self.cp_attn_mask = self._batch_tokenize(
                        cp_list, tokenizer, config.max_cp_length
                    )
            else:
                self.config.use_counter_party = False

        # 5. Partitioning
        logger.info("Computing Window partitions...")
        df['__internal_idx__'] = np.arange(len(df))

        self.window_indices: list[npt.NDArray[np.int64]] = []
        self.window_has_positive_signal = []

        grouped = df.groupby([self.fields.accountId, '__direction__'], sort=False)

        for _, group in grouped:
            grp_indices = group['__internal_idx__'].values
            grp_days = self.all_days[grp_indices]
            grp_cycles = self.all_cycles[grp_indices]

            n_items = len(grp_indices)
            for i in range(0, n_items, config.max_seq_len):
                chunk_indices = grp_indices[i: i + config.max_seq_len]
                chunk_days = grp_days[i: i + config.max_seq_len]

                chunk_cycles = grp_cycles[i: i + config.max_seq_len]
                has_signal = np.any(chunk_cycles > 0)
                self.window_has_positive_signal.append(has_signal)

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
        raw_days = self.all_days[indices]

        if len(raw_days) > 0:
            days_relative = (raw_days - raw_days.min()).astype(np.float32)
        else:
            days_relative = np.array([], dtype=np.float32)

        sample: TransactionSample = {
            # Shared
            "amounts": self.all_log_amounts[indices],
            "days": days_relative,
            "calendar_features": self.all_calendar[indices],
            "pattern_ids": self.all_pattern_ids[indices],
            "cycles": self.all_cycles[indices],
            "original_index": self.all_true_indices[indices],

            # Text Options (One set will be None)
            "text_ids": self.text_input_ids[indices] if self.text_input_ids is not None else None,
            "text_mask": self.text_attn_mask[indices] if self.text_attn_mask is not None else None,
            "cached_text": self.cached_text[indices] if self.cached_text is not None else None,

            # CP Options
            "cp_ids": self.cp_input_ids[indices] if self.cp_input_ids is not None else None,
            "cp_mask": self.cp_attn_mask[indices] if self.cp_attn_mask is not None else None,
            "cached_cp": self.cached_cp[indices] if self.cached_cp is not None else None,
        }
        return sample


def collate_fn(batch: list[TransactionSample], config: MultiExpConfig) -> TransactionBatch:
    # Length determination depends on what data we have
    if config.use_cached_embeddings:
        lengths = [len(x['cached_text']) for x in batch]
    else:
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

    # Shared Tensors
    amounts = collate_tensor("amounts", (1,), torch.float32)
    days = collate_tensor("days", (1,), torch.float32)
    calendar = collate_tensor("calendar_features", (6,), torch.float32)
    cycles = collate_tensor("cycles", (), torch.long)
    pattern_ids = collate_tensor("pattern_ids", (), torch.long, padding_val=-1)
    original_index = collate_tensor("original_index", (), torch.long, padding_val=-1)

    padding_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i in range(batch_size):
        padding_mask[i, :lengths[i]] = True

    result: TransactionBatch = {
        "amounts": amounts,
        "days": days,
        "calendar_features": calendar,
        "adjacency_target": None,
        "cycle_target": cycles,
        "pattern_ids": pattern_ids,
        "padding_mask": padding_mask,
        "original_index": original_index,
        # Defaults
        "input_ids": None, "attention_mask": None,
        "cp_input_ids": None, "cp_attention_mask": None,
        "cached_text": None, "cached_cp": None
    }

    # Conditional Collation
    if config.use_cached_embeddings:
        # Determine embedding dimension from the first sample (safe assumption)
        emb_dim = batch[0]['cached_text'].shape[1]
        result["cached_text"] = collate_tensor("cached_text", (emb_dim,), torch.float32)

        if config.use_counter_party:
            result["cached_cp"] = collate_tensor("cached_cp", (emb_dim,), torch.float32)
    else:
        result["input_ids"] = collate_tensor("text_ids", (config.max_text_length,), torch.long)
        result["attention_mask"] = collate_tensor("text_mask", (config.max_text_length,), torch.long)

        if config.use_counter_party:
            result["cp_input_ids"] = collate_tensor("cp_ids", (config.max_cp_length,), torch.long)
            result["cp_attention_mask"] = collate_tensor("cp_mask", (config.max_cp_length,), torch.long)

    return result