import numpy as np
import pandas as pd
from common.config import FilterConfig


class StabilityValidator:
    """
    Centralizes the logic for deciding if a group of transactions 
    is 'recurring' based on Date and Amount variance.
    """

    def __init__(self, config: FilterConfig):
        self.config = config

    def evaluate(self, dates: pd.Series, amounts: pd.Series) -> bool:
        """Returns True if the group meets all stability criteria."""

        # 1. Count Check
        if len(dates) < self.config.min_txns_for_period:
            return False

        # 2. Amount Stability
        if self.config.stability_metric == 'mad':
            # Robust: Median Absolute Deviation
            amt_median = np.median(amounts)
            amt_var = np.median(np.abs(amounts - amt_median))
        else:
            # Standard: Standard Deviation
            amt_var = amounts.std()

        if amt_var > self.config.amount_variance_threshold:
            return False

        # 3. Date/Cycle Stability
        # Calculate gaps between consecutive transactions
        gaps = dates.sort_values().diff().dt.days.dropna()

        if len(gaps) < (self.config.min_txns_for_period - 1):
            return False

        if self.config.stability_metric == 'mad':
            gap_median = np.median(gaps)
            gap_var = np.median(np.abs(gaps - gap_median))
        else:
            gap_var = gaps.std()

        if gap_var > self.config.date_variance_threshold:
            return False

        return True