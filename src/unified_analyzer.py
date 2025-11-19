import logging
import pandas as pd
import numpy as np
from config import FilterConfig, FieldConfig, ClusteringStrategy
from greedy_analyzer import GreedyGroupAnalyzer
from group_analyzer import RecurringGroupAnalyzer

logger = logging.getLogger(__name__)


class StrategyDispatcher:
    """
    A wrapper that instantiates the correct underlying analyzer
    based on FilterConfig.strategy.
    """

    def __init__(self, filter_config: FilterConfig, field_config: FieldConfig):
        self.config = filter_config
        self.fields = field_config
        self._impl = self._build_impl()

    def _build_impl(self):
        """Factory method to create the specific analyzer instance."""

        if self.config.strategy == ClusteringStrategy.GREEDY:
            logger.info(f"Initializing Greedy Strategy (thresh={self.config.greedy_sim_threshold})")
            return GreedyGroupAnalyzer(
                filter_config=self.config,
                field_config=self.fields,
                # Map config fields to Greedy's specific constructor args
                sim_threshold=self.config.greedy_sim_threshold,
                amount_tol_abs=self.config.greedy_amount_tol_abs,
                amount_tol_pct=self.config.greedy_amount_tol_pct
            )

        elif self.config.strategy == ClusteringStrategy.DBSCAN:
            logger.info(f"Initializing DBSCAN Strategy (eps={self.config.dbscan_eps})")
            # RecurringGroupAnalyzer reads eps/min_samples directly from filter_config
            return RecurringGroupAnalyzer(
                filter_config=self.config,
                field_config=self.fields
            )

        else:
            raise ValueError(f"Unknown clustering strategy: {self.config.strategy}")

    def analyze_account(self, account_df: pd.DataFrame, embeddings: np.ndarray):
        """Delegates to the selected implementation."""
        return self._impl.analyze_account(account_df, embeddings)