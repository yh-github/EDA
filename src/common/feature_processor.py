import logging
from collections import Counter
from dataclasses import fields
from typing import Self
import numpy as np
import pandas as pd
from common.config import TWO_PI, FieldConfig
from common.data import FeatureSet

logger = logging.getLogger(__name__)

from dataclasses import dataclass, field

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

    # This is the change:
    # It maps a feature name (e.g., 'cents_id') to its config
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

    @staticmethod
    def create(params: FeatProcParams, fields_config: FieldConfig = FieldConfig()):
        return HybridFeatureProcessor(fields_config=fields_config, is_nop=params.is_nop(), **params.__dict__)

    def is_nop(self) -> bool:
        return self._is_nop

    def __init__(self,
                 is_nop: bool,
                 k_top: int,
                 n_bins: int,
                 use_cyclical_dates: bool,
                 use_categorical_dates: bool,
                 use_continuous_amount: bool,
                 use_categorical_amount: bool,
                 use_is_positive: bool,
                 use_behavioral_features: bool,
                 fields_config: FieldConfig):
        """
        Initialize the processor.

        :param k_top: How many "magic number" amounts to find (e.g., Top 500).
        :param n_bins: How many "fallback" bins to create for all other amounts.
        :param use_cyclical_dates: (bool) Create sin/cos features for dates.
        :param use_categorical_dates: (bool) Create _id features for date embeddings.
        :param use_continuous_amount: (bool) Create is_positive & log_abs_amount.
        :param use_categorical_amount: (bool) Create amount_token_id (magic/bins).
        """
        self.already_fitted = False
        self._is_nop = is_nop
        self.fields_config = fields_config

        # Store parameters
        self.k_top = k_top
        self.n_bins = n_bins

        # Store ablation flags
        self.use_cyclical_dates = use_cyclical_dates
        self.use_categorical_dates = use_categorical_dates
        self.use_continuous_amount = use_continuous_amount
        self.use_categorical_amount = use_categorical_amount
        self.use_is_positive = use_is_positive
        self.use_behavioral_features = use_behavioral_features

        # These will be "learned" during .fit()
        self.top_k_amounts: set[float] = set()
        self.bin_edges: np.ndarray | None = None
        self.vocab_map: dict[str | float, int] = {}
        self.vocab_size: int = 0
        self.magic_number_cents_distribution: Counter = Counter()

        # Store vocab IDs for health checks
        self.unknown_token_id: int | None = None
        self.top_k_token_ids: set[int] = set()
        self.bin_token_ids: set[int] = set()

        if self.k_top == 0 or self.n_bins == 0 or not self.use_categorical_amount:
            self.k_top = 0
            self.n_bins = 0
            self.use_categorical_amount = False

    def _fit_categorical_amount(self, df: pd.DataFrame):
        amount_col = self.fields_config.amount
        logger.info("Fitting categorical amount features...")
        top_k_series = df[amount_col].value_counts().head(self.k_top).index
        self.top_k_amounts = set(top_k_series)

        # --- Cents Distribution Analysis ---
        self.magic_number_cents_distribution = Counter()
        for amount in self.top_k_amounts:
            cents = int(round(abs(amount) * 100)) % 100
            self.magic_number_cents_distribution[cents] += 1

        # logger.info("--- Magic Number Cents Analysis (Top 10) ---")
        # for cents, count in self.magic_number_cents_distribution.most_common(10):
        #     logger.info(f"  {cents}: {count}")

        # Create the fallback data (all amounts NOT in the Top-K)
        fallback_amounts = df[~df[amount_col].isin(self.top_k_amounts)][amount_col]

        if not fallback_amounts.empty:
            log_abs_fallback = np.log(np.abs(fallback_amounts) + 1)
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            all_bins = np.quantile(log_abs_fallback, quantiles)
            self.bin_edges = np.unique(all_bins)

            if len(self.bin_edges) < 2:
                self.bin_edges = np.array([-np.inf, np.inf])
        else:
            self.bin_edges = np.array([-np.inf, np.inf])

        # --- Build the Amount "Vocabulary" ---
        self.vocab_map = {}
        vocab_id_counter = 0

        self.vocab_map['[PAD]'] = vocab_id_counter
        vocab_id_counter += 1

        self.vocab_map['[UNKNOWN]'] = vocab_id_counter
        self.unknown_token_id = vocab_id_counter
        vocab_id_counter += 1

        self.top_k_token_ids.clear()
        for amount in sorted(list(self.top_k_amounts)):
            self.vocab_map[amount] = vocab_id_counter
            self.top_k_token_ids.add(vocab_id_counter)
            vocab_id_counter += 1

        self.bin_token_ids.clear()
        # Check for valid bin_edges
        if self.bin_edges is not None:
            for i in range(len(self.bin_edges) - 1):
                bin_name = f"log_bin_{i}"
                self.vocab_map[bin_name] = vocab_id_counter
                self.bin_token_ids.add(vocab_id_counter)
                vocab_id_counter += 1

        self.vocab_size = len(self.vocab_map)

        logger.info(f"Fit complete. Found {len(self.top_k_amounts)} magic numbers.")
        bin_count = len(self.bin_edges) - 1 if self.bin_edges is not None else 0
        logger.info(f"Created {bin_count} fallback bins.")
        logger.info(f"Total Amount Vocabulary size: {self.vocab_size}")

    def _build_meta(self) -> FeatureMetadata:
        # --- 1. Create the metadata object we will populate ---
        meta = FeatureMetadata()

        # --- 2. Populate metadata from STATIC params ---
        if self.use_cyclical_dates:
            meta.cyclical_cols.extend([
                'day_of_week_sin', 'day_of_week_cos', 'day_of_month_sin', 'day_of_month_cos',
                'day_of_14_cycle_sin', 'day_of_14_cycle_cos'
            ])

        if self.use_categorical_dates:
            meta.categorical_features['day_of_week_id'] = CategoricalFeatureConfig(
                vocab_size=7, embedding_dim=16
            )
            meta.categorical_features['day_of_month_id'] = CategoricalFeatureConfig(
                vocab_size=31, embedding_dim=32
            )

        if self.use_continuous_amount:
            meta.continuous_scalable_cols.append('log_abs_amount')
            if self.use_is_positive:
                meta.categorical_features['is_positive'] = CategoricalFeatureConfig(
                    vocab_size=2, embedding_dim=2
                )

        # (Example: Add text_length here if self.params.use_text_length)
        # (Example: Add cents_id here if self.params.use_cents)

        # --- 3. Run DYNAMIC fitting logic ---
        if self.use_categorical_amount:
            # Assume self.vocab_map is now built
            self.vocab_size = len(self.vocab_map)
            # add the discovered config to the metadata
            meta.categorical_features['amount_token_id'] = CategoricalFeatureConfig(
                vocab_size=self.vocab_size,
                embedding_dim=64 # proportional?
            )

            logger.info(f"Fit complete. Found {len(self.top_k_amounts)} magic numbers.")

            if self.use_behavioral_features:
                meta.static_real_cols = [
                    'acc_stat_txn_freq',
                    'acc_stat_amount_mean',
                    'acc_stat_amount_std',
                    'acc_stat_amount_max',
                    'acc_stat_amount_median',
                    'acc_stat_amount_q25',
                    'acc_stat_amount_q75',
                    'acc_stat_amount_iqr',
                    'acc_stat_spike_ratio'
                ]

        return meta

    def fit(self, df: pd.DataFrame):
        """
        Learns all the "rules" from the training data based on enabled features.
        """
        if self.already_fitted:
            raise Exception("Already fitted")
        self.already_fitted = True

        logger.info(f"Fitting processor on {len(df)} rows...")

        # --- Learn from 'amount' (if enabled) ---
        if self.use_categorical_amount:
            self._fit_categorical_amount(df)

        return self._build_meta()


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the learned rules to transform a DataFrame based on enabled features.
        """
        logger.info(f"Transforming {len(df)} rows...")

        date_col = self.fields_config.date
        amount_col = self.fields_config.amount

        features = pd.DataFrame(index=df.index)

        # --- 1. Date Features ---
        if self.use_cyclical_dates or self.use_categorical_dates:
            try:
                dates = pd.to_datetime(df[date_col], format='%m/%d/%Y %H:%M:%S')
            except ValueError:
                dates = pd.to_datetime(df[date_col], errors='raise')

            if dates.isnull().any():
                raise ValueError(f"Found {dates.isnull().sum()} rows with missing or invalid dates.")

            day_of_week_raw = dates.dt.dayofweek  # 0=Monday, 6=Sunday
            day_of_month_raw = dates.dt.day  # 1-31

            if self.use_cyclical_dates:
                # days_in_month = dates.dt.days_in_month # TODO additional feature?
                features['day_of_week_sin'] = np.sin(day_of_week_raw * (TWO_PI / 7))
                features['day_of_week_cos'] = np.cos(day_of_week_raw * (TWO_PI / 7))
                features['day_of_month_sin'] = np.sin(day_of_month_raw * (TWO_PI / 31))
                features['day_of_month_cos'] = np.cos(day_of_month_raw * (TWO_PI / 31))

                # 14-day cycle
                epoch_days = (dates - pd.Timestamp("2000-01-01")).dt.days # type: ignore
                raw_cycle_day_14 = epoch_days % 14
                features['day_of_14_cycle_sin'] = np.sin(raw_cycle_day_14 * (TWO_PI / 14))
                features['day_of_14_cycle_cos'] = np.cos(raw_cycle_day_14 * (TWO_PI / 14))

            if self.use_categorical_dates:
                features['day_of_week_id'] = day_of_week_raw
                features['day_of_month_id'] = day_of_month_raw - 1  # 1-31 -> 0-30

        # --- 2. Amount Features ---
        if self.use_continuous_amount:
            features['is_positive'] = (df[amount_col] > 0).astype(int)
            features['log_abs_amount'] = np.log(np.abs(df[amount_col]) + 1)

        if self.use_categorical_amount:
            if not self.vocab_map:
                raise RuntimeError(
                    "Processor has not been fitted, but 'use_categorical_amount' is True. Call .fit() first.")
            features['amount_token_id'] = df[amount_col].apply(self._tokenize_amount)

        # --- 3. Behavioral Features (Refactored) ---
        if self.use_behavioral_features:
            behavioral_features = self._generate_behavioral_features(df)
            # Merge behavioral features into the main dataframe
            features = pd.concat([features, behavioral_features], axis=1)

        logger.info("Transform complete.")
        return features

    def _generate_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates static statistical profiles for each account.
        Returns a DataFrame with the same index as df, containing broadcasted stats.
        """
        acc_col = self.fields_config.accountId
        amt_col = self.fields_config.amount
        date_col = self.fields_config.date

        # Output DataFrame
        beh_df = pd.DataFrame(index=df.index)

        # Create a temporary DataFrame for aggregation
        temp = pd.DataFrame({
            'acc': df[acc_col],
            'amt': df[amt_col],
            'dt': pd.to_datetime(df[date_col])
        })

        # --- A. Frequency Stats ---
        # Calculate "Active Duration" in Days per account
        range_per_acc = temp.groupby('acc')['dt'].transform(lambda x: (x.max() - x.min()).days + 1)
        count_per_acc = temp.groupby('acc')['amt'].transform('count')

        # Transactions per Day (Avoid division by zero/short ranges)
        beh_df['acc_stat_txn_freq'] = count_per_acc / range_per_acc.clip(lower=1)

        # --- B. Basic Amount Stats (Log Space) ---
        log_amt = np.log(np.abs(temp['amt']) + 1)
        beh_df['acc_stat_amount_mean'] = log_amt.groupby(temp['acc']).transform('mean')
        beh_df['acc_stat_amount_std'] = log_amt.groupby(temp['acc']).transform('std').fillna(0)

        # --- C. Advanced Distribution Stats (Raw Space) ---
        # 1. Robust Centrality & Extremes
        beh_df['acc_stat_amount_max'] = temp.groupby('acc')['amt'].transform('max')
        beh_df['acc_stat_amount_median'] = temp.groupby('acc')['amt'].transform('median')

        # 2. Dispersion / IQR
        # Note: using lambda for quantiles in transform can be slow on massive data,
        # but it is the most direct way to broadcast.
        beh_df['acc_stat_amount_q25'] = temp.groupby('acc')['amt'].transform(lambda x: x.quantile(0.25))
        beh_df['acc_stat_amount_q75'] = temp.groupby('acc')['amt'].transform(lambda x: x.quantile(0.75))
        beh_df['acc_stat_amount_iqr'] = beh_df['acc_stat_amount_q75'] - beh_df['acc_stat_amount_q25']

        # 3. "Spikiness" Ratio (Max / Median)
        # Avoid division by zero if median is 0 (e.g. free tier / zero balance checks)
        median_safe = beh_df['acc_stat_amount_median'].replace(0, 1.0)
        beh_df['acc_stat_spike_ratio'] = beh_df['acc_stat_amount_max'] / median_safe

        return beh_df

    def _tokenize_amount(self, amount: float) -> int:
        """Helper function to find the correct token ID for an amount."""

        if self.unknown_token_id is None or self.bin_edges is None:
            raise RuntimeError("Processor has not been fitted. Call .fit() first.")

        if amount in self.top_k_amounts:
            return self.vocab_map[amount]

        log_val = np.log(np.abs(amount) + 1)
        # bin_index = np.digitize(log_val, self.bin_edges) - 1
        bin_index = np.digitize(log_val, self.bin_edges, right=True) - 1

        if bin_index < 0:
            bin_index = 0
        elif bin_index >= len(self.bin_edges) - 1:
            bin_index = len(self.bin_edges) - 2

        bin_name = f"log_bin_{bin_index}"

        if bin_name in self.vocab_map:
            return self.vocab_map[bin_name]
        else:
            return self.unknown_token_id


# ---
# Health Check Utilities
# ---

def check_unknown_rate(
        processor: HybridFeatureProcessor,
        features: pd.DataFrame,
        name: str
) -> dict[str, int | float]:
    """Calculates and prints the [UNKNOWN] token rate."""
    if 'amount_token_id' not in features.columns:
        logger.info(f"\n{name}: Categorical amount feature not enabled. Skipping.")
        return {}

    unknown_id = processor.unknown_token_id
    total_rows = len(features)
    unknown_count = (features['amount_token_id'] == unknown_id).sum()
    unknown_pct = (unknown_count / total_rows) * 100 if total_rows > 0 else 0

    logger.info(f"\n{name} [UNKNOWN] Token Rate:")
    logger.info(f"  {unknown_count} of {total_rows} rows mapped to [UNKNOWN] ({unknown_pct:.2f}%)")

    if unknown_pct > 10:
        logger.warning(f"  **WARNING:** High [UNKNOWN] rate.")
    else:
        logger.info(f"  **INFO:** Low [UNKNOWN] rate. This is good!")

    return {'count': unknown_count, 'total': total_rows, 'percent': unknown_pct}


def check_token_distribution(
        processor: HybridFeatureProcessor,
        features: pd.DataFrame,
        name: str
) -> dict[str, dict[str, int | float]]:
    """Shows how the data was categorized."""
    if 'amount_token_id' not in features.columns:
        logger.info(f"\n{name}: Categorical amount feature not enabled. Skipping.")
        return {}

    logger.info(f"\n{name} Token Distribution:")
    counts:Counter[int] = Counter(features['amount_token_id']) # type: ignore
    total_rows = len(features)

    magic_count = sum(counts.get(i, 0) for i in processor.top_k_token_ids)
    bin_count = sum(counts.get(i, 0) for i in processor.bin_token_ids)
    unknown_count = counts.get(processor.unknown_token_id, 0)

    magic_pct = (magic_count / total_rows) * 100 if total_rows > 0 else 0
    bin_pct = (bin_count / total_rows) * 100 if total_rows > 0 else 0
    unknown_pct = (unknown_count / total_rows) * 100 if total_rows > 0 else 0

    logger.info(f"  - Mapped to 'Magic Numbers': {magic_count} rows ({magic_pct:.2f}%)")
    logger.info(f"  - Mapped to 'Fallback Bins': {bin_count} rows ({bin_pct:.2f}%)")
    logger.info(f"  - Mapped to '[UNKNOWN]':    {unknown_count} rows ({unknown_pct:.2f}%)")

    stats = {
        'magic': {'count': magic_count, 'percent': magic_pct},
        'bin': {'count': bin_count, 'percent': bin_pct},
        'unknown': {'count': unknown_count, 'percent': unknown_pct}
    }
    return stats


@dataclass(frozen=True)
class FeatureHyperParams:
    # feature-related parameters for initializing the HybridModel.
    text_embed_dim: int = 0
    continuous_feat_dim: int = 0
    categorical_vocab_sizes: dict[str, int] = field(default_factory=dict)
    embedding_dims: dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        train_features: FeatureSet,
        metadata: FeatureMetadata
    ) -> Self:
        """
        Dynamically builds the HybridModelConfig dataclass based on the
        feature metadata returned from the processor.
        """

        # 1. Get dimensions for text and continuous features
        text_embed_dim = train_features.X_text.shape[1]
        continuous_feat_dim = train_features.X_continuous.shape[1]

        # 2. Get categorical config directly FROM THE METADATA
        categorical_vocab_sizes = {
            name: config.vocab_size
            for name, config in metadata.categorical_features.items()
        }

        # 3. Get embedding dims directly FROM THE METADATA
        embedding_dims = {
            name: config.embedding_dim
            for name, config in metadata.categorical_features.items()
        }

        # 4. Return the complete, frozen config object
        return cls(
            text_embed_dim=text_embed_dim,
            continuous_feat_dim=continuous_feat_dim,
            categorical_vocab_sizes=categorical_vocab_sizes,
            embedding_dims=embedding_dims
        )

