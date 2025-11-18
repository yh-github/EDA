import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum, auto
from sklearn.cluster import DBSCAN

# Config imports
from config import FilterConfig, FieldConfig

# --- Optional SOTA Imports ---
import umap
import hdbscan

logger = logging.getLogger(__name__)


class PatternStatus(Enum):
    FOUND = auto()
    NO_PATTERN_NOISE = auto()
    INSUFFICIENT_DATA = auto()


@dataclass(frozen=True, slots=True)
class RecurringPattern:
    """
    Describes a 'Recurring Pattern' found in an account.
    Uses modern Python 3.10+ slots for memory efficiency.
    """
    cluster_id: int
    status: PatternStatus
    transaction_ids: list[str] = field(default_factory=list)

    # Pattern Metadata
    counterparty_description: str | None = None

    # Value Stats (Robust)
    amount_median: float | None = None
    amount_mad: float | None = None  # Median Absolute Deviation

    # Periodicity Stats (Robust)
    cycle_days_median: float | None = None
    cycle_days_mad: float | None = None

    @property
    def is_highly_regular(self) -> bool:
        """
        Returns True if the pattern is extremely stable (low jitter).
        """
        if self.status != PatternStatus.FOUND:
            return False
        # Allow max 1 day jitter and very low amount variance
        return (self.cycle_days_mad or 0.0) < 1.0 and (self.amount_mad or 0.0) < 0.10


class RecurringPatternExtractor:
    """
    Extracts recurring patterns using density-based clustering and robust statistics.
    """

    def __init__(self,
                 filter_config: FilterConfig,
                 field_config: FieldConfig,
                 use_advanced_clustering: bool = True):

        self.config = filter_config
        self.fields = field_config
        self.use_sota = use_advanced_clustering and HAS_SOTA_LIBS

        if use_advanced_clustering and not HAS_SOTA_LIBS:
            logger.warning(
                "Advanced clustering requested but 'umap-learn' or 'hdbscan' not found. Falling back to sklearn DBSCAN.")

    def extract_patterns(self, account_df: pd.DataFrame, embeddings: np.ndarray) -> list[RecurringPattern]:
        if account_df.empty or len(embeddings) == 0:
            return []

        # --- Step 1: Advanced Clustering ---
        labels = self._cluster_embeddings(embeddings)

        # --- Step 2: Pattern Analysis ---
        patterns: list[RecurringPattern] = []
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:
                continue  # Noise

            # Filter DataFrame for this cluster
            cluster_mask = (labels == label)
            group_df = account_df[cluster_mask]

            pattern = self._analyze_group_robust(group_df, int(label))
            patterns.append(pattern)

        return patterns

    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Applies UMAP dimensionality reduction followed by HDBSCAN (if available),
        or standard DBSCAN on raw embeddings.
        """
        if self.use_sota:
            # 1. Reduce dims (768 -> 10) to avoid curse of dimensionality
            # n_neighbors=15 preserves local structure (who is similar to who)
            reducer = umap.UMAP(n_neighbors=15, n_components=10, metric='cosine', random_state=42)
            reduced_emb = reducer.fit_transform(embeddings)

            # 2. Cluster with HDBSCAN (No fixed epsilon required)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=self.config.dbscan_min_samples, metric='euclidean')
            return clusterer.fit_predict(reduced_emb)
        else:
            # Fallback: Standard DBSCAN on raw embeddings
            dbscan = DBSCAN(
                eps=self.config.dbscan_eps,
                min_samples=self.config.dbscan_min_samples,
                metric='cosine'
            )
            return dbscan.fit_predict(embeddings)

    def _analyze_group_robust(self, df: pd.DataFrame, cluster_id: int) -> RecurringPattern:
        """
        Uses Median Absolute Deviation (MAD) for robust pattern detection.
        MAD is resilient to occasional outliers (late payments, random fees).
        """
        t_ids = df[self.fields.trId].tolist()

        if len(df) < self.config.min_txns_for_period:
            return RecurringPattern(cluster_id, PatternStatus.INSUFFICIENT_DATA, t_ids)

        # --- Robust Amount Stats ---
        amounts = df[self.fields.amount].values
        amt_median = float(np.median(amounts))
        amt_mad = float(np.median(np.abs(amounts - amt_median)))

        # Threshold check (using config std threshold as a proxy for MAD allowance)
        if amt_mad > self.config.amount_std_threshold:
            return RecurringPattern(cluster_id, PatternStatus.NO_PATTERN_NOISE, t_ids)

        # --- Robust Date Stats ---
        try:
            dates = pd.to_datetime(df[self.fields.date])
            dates_sorted = dates.sort_values()
            deltas_days = dates_sorted.diff().dt.days.dropna().values

            if len(deltas_days) == 0:
                return RecurringPattern(cluster_id, PatternStatus.INSUFFICIENT_DATA, t_ids)

            cycle_median = float(np.median(deltas_days))
            cycle_mad = float(np.median(np.abs(deltas_days - cycle_median)))

            # "Strict" check: If the cycle jitters by more than ~2 days (MAD), reject it.
            # We use date_std_threshold as the limit.
            if cycle_mad > self.config.date_std_threshold:
                return RecurringPattern(cluster_id, PatternStatus.NO_PATTERN_NOISE, t_ids)

            # --- Success ---
            # Use the most frequent text description
            best_desc = df[self.fields.text].mode()[0]

            return RecurringPattern(
                cluster_id=cluster_id,
                status=PatternStatus.FOUND,
                transaction_ids=t_ids,
                counterparty_description=best_desc,
                amount_median=amt_median,
                amount_mad=amt_mad,
                cycle_days_median=cycle_median,
                cycle_days_mad=cycle_mad
            )

        except Exception as e:
            logger.warning(f"Error analyzing cluster {cluster_id}: {e}")
            return RecurringPattern(cluster_id, PatternStatus.NO_PATTERN_NOISE, t_ids)