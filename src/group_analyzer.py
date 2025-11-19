import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Optional, List
from config import FilterConfig, FieldConfig


class GroupStabilityStatus(Enum):
    STABLE = auto()
    ERROR_UNKNOWN = auto()
    NOT_STABLE_INSUFFICIENT_TXNS = auto()
    NOT_STABLE_AMOUNT_VARIANCE = auto()
    NOT_STABLE_DATE_VARIANCE = auto()


@dataclass(frozen=True)
class RecurringGroupResult:
    """
    Holds the full analysis for a single discovered group of
    recurring transactions.
    """
    cluster_id: int
    status: GroupStabilityStatus

    # List of transaction IDs from the original DataFrame
    transaction_ids: List[str] = field(default_factory=list)

    # Analysis results
    cycle_days: Optional[float] = None
    predicted_amount: Optional[float] = None
    next_predicted_date: Optional[datetime] = None

    # Raw stats for inspection
    amount_mean: Optional[float] = None
    amount_std: Optional[float] = None
    date_delta_mean_days: Optional[float] = None
    date_delta_std_days: Optional[float] = None


from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger(__name__)


class RecurringGroupAnalyzer:
    """
    Analyzes a set of recurring transactions for a single account
    to cluster them and forecast the next event for each group.
    """

    def __init__(self, filter_config: FilterConfig, field_config: FieldConfig):
        """
        Initializes the analyzer with the stability rules.

        :param filter_config: A FilterConfig dataclass with analysis rules.
        :param field_config: A FieldConfig dataclass with column names.
        """
        self.config = filter_config
        self.fields = field_config

        # Initialize the DBSCAN clusterer from config
        # Use default eps=0.5, min=2 if not specified in config (backward compatibility)
        eps = getattr(self.config, 'dbscan_eps', 0.5)
        min_samples = getattr(self.config, 'dbscan_min_samples', 2)

        self.dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="cosine"
        )

    def analyze_account(
            self,
            account_df: pd.DataFrame,
            embeddings: np.ndarray
    ) -> List[RecurringGroupResult]:
        """
        Analyzes a single account's recurring transactions and their embeddings.
        """
        if account_df.empty or embeddings.shape[0] == 0:
            return []

        if len(account_df) != embeddings.shape[0]:
            raise ValueError(
                f"DataFrame rows ({len(account_df)}) do not match "
                f"embedding rows ({embeddings.shape[0]})"
            )

        # 1. Cluster transactions based on text embeddings
        try:
            cluster_labels = self.dbscan.fit_predict(embeddings)
            account_df = account_df.copy()
            account_df['cluster_id'] = cluster_labels
        except Exception as e:
            logger.error(f"DBSCAN clustering failed: {e}")
            return [RecurringGroupResult(
                cluster_id=-1,
                status=GroupStabilityStatus.ERROR_UNKNOWN
            )]

        results = []

        # 2. Iterate over each found cluster (skip -1, which is "noise")
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:
                continue  # Skip noise points

            group_df = account_df[account_df['cluster_id'] == cluster_id].copy()

            # 3. Analyze the stability and forecast for this specific group
            group_result = self._analyze_group(group_df, int(cluster_id))
            results.append(group_result)

        return results

    def _analyze_group(
            self,
            group_df: pd.DataFrame,
            cluster_id: int
    ) -> RecurringGroupResult:
        """
        Analyzes a single cluster of transactions for stability and
        forecasts the next event.
        """
        t_ids = group_df[self.fields.trId].tolist()

        # --- 1. Check Transaction Count ---
        if len(group_df) < self.config.min_txns_for_period:
            return RecurringGroupResult(
                cluster_id=cluster_id,
                status=GroupStabilityStatus.NOT_STABLE_INSUFFICIENT_TXNS,
                transaction_ids=t_ids
            )

        # --- 2. Analyze Amount Stability ---
        amounts = group_df[self.fields.amount].values

        # Calculate variance based on metric (std or mad)
        if self.config.stability_metric == 'mad':
            amount_median = float(np.median(amounts))
            amount_var = float(np.median(np.abs(amounts - amount_median)))
        else:
            amount_var = float(amounts.std())

        amount_mean = float(amounts.mean())
        amount_std_report = float(amounts.std())  # Keep std for reporting consistency

        if amount_var > self.config.amount_variance_threshold:
            return RecurringGroupResult(
                cluster_id=cluster_id,
                status=GroupStabilityStatus.NOT_STABLE_AMOUNT_VARIANCE,
                transaction_ids=t_ids,
                amount_mean=amount_mean,
                amount_std=amount_std_report
            )

        # --- 3. Analyze Date/Period Stability ---
        try:
            dates = pd.to_datetime(group_df[self.fields.date], format='%m/%d/%Y %H:%M:%S')
            if dates.isnull().all():  # Fallback if format doesn't match
                dates = pd.to_datetime(group_df[self.fields.date])

            dates_sorted = dates.sort_values()
            date_deltas_days = dates_sorted.diff().dt.days.dropna()

            if len(date_deltas_days) < (self.config.min_txns_for_period - 1):
                return RecurringGroupResult(
                    cluster_id=cluster_id,
                    status=GroupStabilityStatus.NOT_STABLE_INSUFFICIENT_TXNS,
                    transaction_ids=t_ids
                )

            gaps = date_deltas_days.values

            # Calculate date variance based on metric
            if self.config.stability_metric == 'mad':
                gap_median = float(np.median(gaps))
                date_var = float(np.median(np.abs(gaps - gap_median)))
            else:
                date_var = float(gaps.std())
                if np.isnan(date_var): date_var = 0.0

            date_delta_mean = float(gaps.mean())
            date_delta_std_report = float(gaps.std()) if len(gaps) > 1 else 0.0

            if date_var > self.config.date_variance_threshold:
                return RecurringGroupResult(
                    cluster_id=cluster_id,
                    status=GroupStabilityStatus.NOT_STABLE_DATE_VARIANCE,
                    transaction_ids=t_ids,
                    amount_mean=amount_mean,
                    amount_std=amount_std_report,
                    date_delta_mean_days=date_delta_mean,
                    date_delta_std_days=date_delta_std_report
                )

            # --- 4. SUCCESS: Group is Stable! ---
            last_txn_date = dates_sorted.iloc[-1]
            # Use median gap for prediction if using MAD, else mean
            cycle_days = float(np.median(gaps)) if self.config.stability_metric == 'mad' else date_delta_mean

            next_date = last_txn_date + timedelta(days=cycle_days)

            return RecurringGroupResult(
                cluster_id=cluster_id,
                status=GroupStabilityStatus.STABLE,
                transaction_ids=t_ids,
                cycle_days=cycle_days,
                predicted_amount=amount_mean,
                next_predicted_date=next_date,
                amount_mean=amount_mean,
                amount_std=amount_std_report,
                date_delta_mean_days=date_delta_mean,
                date_delta_std_days=date_delta_std_report
            )

        except Exception as e:
            logger.warning(f"Failed to analyze group {cluster_id}: {e}", exc_info=True)
            return RecurringGroupResult(
                cluster_id=cluster_id,
                status=GroupStabilityStatus.ERROR_UNKNOWN,
                transaction_ids=t_ids
            )