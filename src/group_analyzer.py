import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Optional, List
from config import FilterConfig, FieldConfig


# We use an Enum to clearly state the result of the analysis
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


# In the same file (src/group_analyzer.py)
# Make sure to install sklearn: pip install scikit-learn
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
        self.dbscan = DBSCAN(
            eps=self.config.dbscan_eps,
            min_samples=self.config.dbscan_min_samples,
            metric="cosine"  # Cosine distance is good for high-dim embedding vectors
        )

    def analyze_account(
            self,
            account_df: pd.DataFrame,
            embeddings: np.ndarray
    ) -> List[RecurringGroupResult]:
        """
        Analyzes a single account's recurring transactions and their embeddings.

        :param account_df: A DataFrame filtered to ONE account and
                           ONLY recurring transactions.
        :param embeddings: A NumPy array of embeddings corresponding
                           to each row in account_df.
        :return: A list of RecurringGroupResult objects.
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
            group_result = self._analyze_group(group_df, cluster_id)
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

        # --- 1. Check Transaction Count ---
        if len(group_df) < self.config.min_txns_for_period:
            return RecurringGroupResult(
                cluster_id=cluster_id,
                status=GroupStabilityStatus.NOT_STABLE_INSUFFICIENT_TXNS,
                transaction_ids=group_df[self.fields.trId].tolist()
            )

        # --- 2. Analyze Amount Stability ---
        amounts = group_df[self.fields.amount]
        amount_std = amounts.std()
        amount_mean = amounts.mean()

        if amount_std > self.config.amount_std_threshold:
            return RecurringGroupResult(
                cluster_id=cluster_id,
                status=GroupStabilityStatus.NOT_STABLE_AMOUNT_VARIANCE,
                transaction_ids=group_df[self.fields.trId].tolist(),
                amount_mean=amount_mean,
                amount_std=amount_std
            )

        # --- 3. Analyze Date/Period Stability ---
        # Convert to datetime, sort, and calculate time deltas
        try:
            # Assuming your date format from config.py
            dates = pd.to_datetime(group_df[self.fields.date], format='%m/%d/%Y %H:%M:%S')
            dates_sorted = dates.sort_values()

            # .diff() calculates difference from previous row
            # .dt.days converts timedelta to a float number of days
            date_deltas_days = dates_sorted.diff().dt.days.dropna()

            # We need at least 2 deltas (3 txns) to get a standard deviation
            if len(date_deltas_days) < 2:
                return RecurringGroupResult(
                    cluster_id=cluster_id,
                    status=GroupStabilityStatus.NOT_STABLE_INSUFFICIENT_TXNS,
                    transaction_ids=group_df[self.fields.trId].tolist()
                )

            date_delta_std = date_deltas_days.std()
            date_delta_mean = date_deltas_days.mean()

            if date_delta_std > self.config.date_std_threshold:
                return RecurringGroupResult(
                    cluster_id=cluster_id,
                    status=GroupStabilityStatus.NOT_STABLE_DATE_VARIANCE,
                    transaction_ids=group_df[self.fields.trId].tolist(),
                    amount_mean=amount_mean,
                    amount_std=amount_std,
                    date_delta_mean_days=date_delta_mean,
                    date_delta_std_days=date_delta_std
                )

            # --- 4. SUCCESS: Group is Stable! ---
            last_txn_date = dates_sorted.iloc[-1]
            next_date = last_txn_date + timedelta(days=date_delta_mean)

            return RecurringGroupResult(
                cluster_id=cluster_id,
                status=GroupStabilityStatus.STABLE,
                transaction_ids=group_df[self.fields.trId].tolist(),
                cycle_days=date_delta_mean,
                predicted_amount=amount_mean,
                next_predicted_date=next_date,
                amount_mean=amount_mean,
                amount_std=amount_std,
                date_delta_mean_days=date_delta_mean,
                date_delta_std_days=date_delta_std
            )

        except Exception as e:
            logger.warning(f"Failed to analyze group {cluster_id}: {e}")
            return RecurringGroupResult(
                cluster_id=cluster_id,
                status=GroupStabilityStatus.ERROR_UNKNOWN,
                transaction_ids=group_df[self.fields.trId].tolist()
            )