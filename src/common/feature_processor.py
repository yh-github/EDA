import logging
from collections import Counter
from dataclasses import dataclass, field, fields, replace
from typing import Self

import numpy as np
import pandas as pd

from common.config import TWO_PI, FieldConfig
from common.data import FeatureSet

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CategoricalFeatureConfig:
    """Holds the config for a single categorical feature."""
    vocab_size: int
    embedding_dim: int


@dataclass
class FeatureMetadata:
    """
    A single object to hold all metadata about the generated features.
    """
    cyclical_cols: list[str] = field(default_factory=list)
    continuous_scalable_cols: list[str] = field(default_factory=list)
    categorical_features: dict[str, CategoricalFeatureConfig] = field(default_factory=dict)
    static_real_cols: list[str] = field(default_factory=list)


@dataclass
class FeatProcParams:
    use_cyclical_dates: bool = True
    use_categorical_dates: bool = True
    use_continuous_amount: bool = True
    use_is_positive: bool = False
    use_categorical_amount: bool = False
    k_top: int = 20
    n_bins: int = 20
    use_behavioral_features: bool = False
    text_dim_reduce: int|None = None

    def is_nop(self) -> bool:
        return all(
            not getattr(self, f.name)
            for f in fields(self)
            if f.name.startswith("use_") and f.type == bool
        )

    @classmethod
    def all_off(cls) -> Self:
        kwargs = {
            f.name: False
            for f in fields(cls)
            if f.name.startswith("use_") and f.type == bool
        }
        return cls(**kwargs)


class HybridFeatureProcessor:
    """
    This class takes a raw DataFrame with 'date' and 'amount' columns
    and engineers all the "smart" features we designed.

    It is built to be "fit" on a training set and then "transform"
    any new data, ensuring all rules are consistent.
    """

    @classmethod
    def create(cls, params: FeatProcParams, fields_config: FieldConfig = FieldConfig()) -> Self:
        """Factory method - delegates to __init__."""
        return cls(params, fields_config)

    def __init__(self, params: FeatProcParams, fields_config: FieldConfig = FieldConfig()):
        self.fields_config: FieldConfig = fields_config

        # Keep a sanitized copy of params to avoid side effects on the input object
        self.params: FeatProcParams = replace(params)

        self.already_fitted: bool = False

        # State initialization
        self.top_k_amounts: set[float] = set()
        self.bin_edges: np.ndarray | None = None
        self.vocab_map: dict[str | float, int] = {}
        self.vocab_size: int = 0
        self.magic_number_cents_distribution: Counter[int] = Counter()

        # Token IDs
        self.unknown_token_id: int | None = None
        self.top_k_token_ids: set[int] = set()
        self.bin_token_ids: set[int] = set()

        # Validation / Sanitization Logic
        # If categorical amount is requested but config is invalid (k=0, bins=0), disable it.
        # Conversely, if categorical amount is OFF, force k/bins to 0 for consistency.
        if self.params.k_top == 0 or self.params.n_bins == 0 or not self.params.use_categorical_amount:
            self.params.k_top = 0
            self.params.n_bins = 0
            self.params.use_categorical_amount = False

    def fit(self, df: pd.DataFrame) -> FeatureMetadata:
        """Learns feature distributions (vocabularies, bins) from training data."""
        if self.already_fitted:
            raise RuntimeError("Processor is already fitted.")

        self.already_fitted = True
        logger.info(f"Fitting processor on {len(df)} rows...")

        if self.params.use_categorical_amount:
            self._fit_categorical_amount(df)

        return self._build_meta()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies transformations to create feature vectors."""
        logger.info(f"Transforming {len(df)} rows...")
        features = pd.DataFrame(index=df.index)

        self._transform_dates(df, features)
        self._transform_amounts_continuous(df, features)
        self._transform_amounts_categorical(df, features)

        if self.params.use_behavioral_features:
            behavioral_features = self._generate_behavioral_features(df)
            features = pd.concat([features, behavioral_features], axis=1)

        logger.info("Transform complete.")
        return features

    # --- Internal Fitting Methods ---

    def _fit_categorical_amount(self, df: pd.DataFrame) -> None:
        amount_col = self.fields_config.amount
        logger.info("Fitting categorical amount features...")

        # 1. Identify Magic Numbers
        top_k_series = df[amount_col].value_counts().head(self.params.k_top).index
        self.top_k_amounts = set(top_k_series)

        # 2. Cents Analysis
        for amount in self.top_k_amounts:
            cents = int(round(abs(amount) * 100)) % 100
            self.magic_number_cents_distribution[cents] += 1

        # 3. Binning Fallback
        fallback_amounts = df[~df[amount_col].isin(self.top_k_amounts)][amount_col]
        self.bin_edges = self._calculate_bin_edges(fallback_amounts)

        # 4. Build Vocabulary
        self._build_vocab_map()

    def _calculate_bin_edges(self, series: pd.Series) -> np.ndarray:
        if series.empty:
            return np.array([-np.inf, np.inf])

        log_abs_fallback = np.log(np.abs(series) + 1)
        quantiles = np.linspace(0, 1, self.params.n_bins + 1)
        # Type ignore: quantile returns generic, we know it's array-like
        all_bins = np.quantile(log_abs_fallback, quantiles)
        unique_bins = np.unique(all_bins)

        if len(unique_bins) < 2:
            return np.array([-np.inf, np.inf])
        return unique_bins

    def _build_vocab_map(self) -> None:
        vocab_id_counter = 0
        self.vocab_map = {}

        # Special Tokens
        self.vocab_map['[PAD]'] = vocab_id_counter
        vocab_id_counter += 1

        self.vocab_map['[UNKNOWN]'] = vocab_id_counter
        self.unknown_token_id = vocab_id_counter
        vocab_id_counter += 1

        # Magic Numbers
        self.top_k_token_ids.clear()
        for amount in sorted(list(self.top_k_amounts)):
            self.vocab_map[amount] = vocab_id_counter
            self.top_k_token_ids.add(vocab_id_counter)
            vocab_id_counter += 1

        # Bins
        self.bin_token_ids.clear()
        if self.bin_edges is not None:
            for i in range(len(self.bin_edges) - 1):
                bin_name = f"log_bin_{i}"
                self.vocab_map[bin_name] = vocab_id_counter
                self.bin_token_ids.add(vocab_id_counter)
                vocab_id_counter += 1

        self.vocab_size = len(self.vocab_map)
        logger.info(
            f"Vocab size: {self.vocab_size} ({len(self.top_k_amounts)} magic, {len(self.bin_edges) - 1 if self.bin_edges is not None else 0} bins)")

    def _build_meta(self) -> FeatureMetadata:
        meta = FeatureMetadata()

        if self.params.use_cyclical_dates:
            meta.cyclical_cols.extend([
                'day_of_week_sin', 'day_of_week_cos', 'day_of_month_sin', 'day_of_month_cos',
                'day_of_14_cycle_sin', 'day_of_14_cycle_cos'
            ])

        if self.params.use_categorical_dates:
            meta.categorical_features['day_of_week_id'] = CategoricalFeatureConfig(vocab_size=7, embedding_dim=16)
            meta.categorical_features['day_of_month_id'] = CategoricalFeatureConfig(vocab_size=31, embedding_dim=32)

        if self.params.use_continuous_amount:
            meta.continuous_scalable_cols.append('log_abs_amount')
            if self.params.use_is_positive:
                meta.categorical_features['is_positive'] = CategoricalFeatureConfig(vocab_size=2, embedding_dim=2)

        if self.params.use_categorical_amount:
            meta.categorical_features['amount_token_id'] = CategoricalFeatureConfig(
                vocab_size=self.vocab_size,
                embedding_dim=64
            )

        if self.params.use_behavioral_features:
            meta.static_real_cols = [
                'acc_stat_txn_freq', 'acc_stat_amount_mean', 'acc_stat_amount_std',
                'acc_stat_amount_max', 'acc_stat_amount_median', 'acc_stat_amount_q25',
                'acc_stat_amount_q75', 'acc_stat_amount_iqr', 'acc_stat_spike_ratio'
            ]

        return meta

    # --- Internal Transformation Methods ---

    def _transform_dates(self, df: pd.DataFrame, features: pd.DataFrame) -> None:
        if not (self.params.use_cyclical_dates or self.params.use_categorical_dates):
            return

        date_col = self.fields_config.date
        # Coerce dates safely
        try:
            dates = pd.to_datetime(df[date_col], format='%m/%d/%Y %H:%M:%S')
        except ValueError:
            dates = pd.to_datetime(df[date_col], errors='coerce')

        # Fill NaNs with a default timestamp if necessary, or let downstream handle
        if dates.isnull().any():
            # Basic handling for demo purposes; production code might drop rows
            dates = dates.fillna(pd.Timestamp("1970-01-01"))

        day_of_week_raw = dates.dt.dayofweek
        day_of_month_raw = dates.dt.day

        if self.params.use_cyclical_dates:
            features['day_of_week_sin'] = np.sin(day_of_week_raw * (TWO_PI / 7))
            features['day_of_week_cos'] = np.cos(day_of_week_raw * (TWO_PI / 7))
            features['day_of_month_sin'] = np.sin(day_of_month_raw * (TWO_PI / 31))
            features['day_of_month_cos'] = np.cos(day_of_month_raw * (TWO_PI / 31))

            epoch_days = (dates - pd.Timestamp("2000-01-01")).dt.days
            raw_cycle_day_14 = epoch_days % 14
            features['day_of_14_cycle_sin'] = np.sin(raw_cycle_day_14 * (TWO_PI / 14))
            features['day_of_14_cycle_cos'] = np.cos(raw_cycle_day_14 * (TWO_PI / 14))

        if self.params.use_categorical_dates:
            features['day_of_week_id'] = day_of_week_raw
            features['day_of_month_id'] = day_of_month_raw - 1

    def _transform_amounts_continuous(self, df: pd.DataFrame, features: pd.DataFrame) -> None:
        if self.params.use_continuous_amount:
            amt = df[self.fields_config.amount]
            features['log_abs_amount'] = np.log(np.abs(amt) + 1)
            if self.params.use_is_positive:
                features['is_positive'] = (amt > 0).astype(int)

    def _transform_amounts_categorical(self, df: pd.DataFrame, features: pd.DataFrame) -> None:
        if self.params.use_categorical_amount:
            if not self.vocab_map:
                raise RuntimeError("Processor not fitted, but categorical amount requested.")

            features['amount_token_id'] = df[self.fields_config.amount].apply(self._tokenize_amount)

    def _tokenize_amount(self, amount: float) -> int:
        """Helper to find token ID."""
        if self.unknown_token_id is None or self.bin_edges is None:
            return 0  # Should ideally raise, but runtime safety prefers robust return

        if amount in self.top_k_amounts:
            return self.vocab_map[amount]

        log_val = np.log(np.abs(amount) + 1)
        bin_index = np.digitize(log_val, self.bin_edges, right=True) - 1

        # Clamp index
        bin_index = max(0, min(bin_index, len(self.bin_edges) - 2))

        bin_name = f"log_bin_{bin_index}"
        return self.vocab_map.get(bin_name, self.unknown_token_id)

    def _generate_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized generation of account-level statistics."""
        acc_col = self.fields_config.accountId
        amt_col = self.fields_config.amount
        date_col = self.fields_config.date

        # Ensure we have datetime
        dates = pd.to_datetime(df[date_col], errors='coerce').fillna(pd.Timestamp("1970-01-01"))

        # Groupby is expensive, so we do it once per required aggregation
        # We use 'transform' to broadcast results back to original shape directly
        grouped_amt = df.groupby(acc_col)[amt_col]
        grouped_date = dates.groupby(df[acc_col])

        beh_df = pd.DataFrame(index=df.index)

        # Freq
        # (max - min) in days
        duration = grouped_date.transform(lambda x: (x.max() - x.min()).days + 1)
        count = grouped_amt.transform('count')
        beh_df['acc_stat_txn_freq'] = count / duration.clip(lower=1)

        # Log Amount Stats
        log_amt = np.log(np.abs(df[amt_col]) + 1)
        grouped_log_amt = log_amt.groupby(df[acc_col])
        beh_df['acc_stat_amount_mean'] = grouped_log_amt.transform('mean')
        beh_df['acc_stat_amount_std'] = grouped_log_amt.transform('std').fillna(0)

        # Raw Stats
        beh_df['acc_stat_amount_max'] = grouped_amt.transform('max')
        median = grouped_amt.transform('median')
        beh_df['acc_stat_amount_median'] = median

        # Quantiles (Slow path, but accurate)
        beh_df['acc_stat_amount_q25'] = grouped_amt.transform(lambda x: x.quantile(0.25))
        beh_df['acc_stat_amount_q75'] = grouped_amt.transform(lambda x: x.quantile(0.75))
        beh_df['acc_stat_amount_iqr'] = beh_df['acc_stat_amount_q75'] - beh_df['acc_stat_amount_q25']

        beh_df['acc_stat_spike_ratio'] = beh_df['acc_stat_amount_max'] / median.replace(0, 1.0)

        return beh_df


@dataclass(frozen=True)
class FeatureHyperParams:
    text_embed_dim: int = 0
    continuous_feat_dim: int = 0
    categorical_vocab_sizes: dict[str, int] = field(default_factory=dict)
    embedding_dims: dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(cls, train_features: FeatureSet, metadata: FeatureMetadata) -> Self:
        return cls(
            text_embed_dim=train_features.X_text.shape[1],
            continuous_feat_dim=train_features.X_continuous.shape[1],
            categorical_vocab_sizes={n: c.vocab_size for n, c in metadata.categorical_features.items()},
            embedding_dims={n: c.embedding_dim for n, c in metadata.categorical_features.items()}
        )