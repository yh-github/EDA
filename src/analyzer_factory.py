from config import FilterConfig, ClusteringStrategy
from greedy_analyzer import GreedyGroupAnalyzer
from group_analyzer import RecurringGroupAnalyzer

# from pattern_finder import RecurringPatternExtractor (if you align the outputs)

class UniversalAnalyzer:
    """
    A wrapper that delegates to the specific strategy defined in config.
    """

    def __init__(self, filter_config: FilterConfig, field_config):
        self.config = filter_config
        self.fields = field_config

        if self.config.strategy == ClusteringStrategy.GREEDY:
            self._impl = GreedyGroupAnalyzer(
                filter_config, field_config,
                sim_threshold=filter_config.greedy_sim_threshold,
                # ... pass other greedy specific params
            )
        elif self.config.strategy == ClusteringStrategy.DBSCAN:
            self._impl = RecurringGroupAnalyzer(
                filter_config, field_config
            )
            # You would need to update GroupAnalyzer to accept eps from config inside __init__
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

    def analyze_account(self, account_df, embeddings):
        # Standardize interface
        return self._impl.analyze_account(account_df, embeddings)