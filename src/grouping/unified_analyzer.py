import logging
import pandas as pd
import numpy as np
from common.config import FilterConfig, FieldConfig, ClusteringStrategy
from greedy_analyzer import GreedyGroupAnalyzer
from group_analyzer import RecurringGroupAnalyzer

logger = logging.getLogger(__name__)


class StrategyDispatcher:
    def __init__(self, filter_config: FilterConfig, field_config: FieldConfig):
        self.config = filter_config
        self.fields = field_config
        self._impl = self._build_impl()

    def _build_impl(self):
        if self.config.strategy == ClusteringStrategy.GREEDY:
            logger.info(f"Init Semantic Strategy (thresh={self.config.greedy_sim_threshold})")
            return GreedyGroupAnalyzer(
                filter_config=self.config,
                field_config=self.fields,
                sim_threshold=self.config.greedy_sim_threshold,
                amount_tol_abs=self.config.greedy_amount_tol_abs,
                amount_tol_pct=self.config.greedy_amount_tol_pct,
                metric="cosine"
            )

        elif self.config.strategy == ClusteringStrategy.LEXICAL:
            logger.info(f"Init Lexical Strategy (thresh={self.config.lexical_sim_threshold})")
            # We reuse the Greedy Analyzer but swap the metric
            return GreedyGroupAnalyzer(
                filter_config=self.config,
                field_config=self.fields,
                sim_threshold=self.config.lexical_sim_threshold,  # Use lexical threshold
                amount_tol_abs=self.config.greedy_amount_tol_abs,
                amount_tol_pct=self.config.greedy_amount_tol_pct,
                metric="lexical"
            )

        elif self.config.strategy == ClusteringStrategy.DBSCAN:
            logger.info(f"Init DBSCAN Strategy (eps={self.config.dbscan_eps})")
            return RecurringGroupAnalyzer(
                filter_config=self.config,
                field_config=self.fields
            )

        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

    def analyze_account(self, account_df: pd.DataFrame, embeddings: np.ndarray | None):
        return self._impl.analyze_account(account_df, embeddings)