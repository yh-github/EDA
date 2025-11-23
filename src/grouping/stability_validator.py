import numpy as np
import pandas as pd

class StabilityValidator:
    def __init__(self, config):
        self.config = config

    def is_stable(self, dates: pd.Series, amounts: pd.Series) -> bool:
        # 1. Check count
        if len(dates) < self.config.min_txns_for_period:
            return False

        # 2. Check Amount Stability
        if self.config.stability_metric == 'mad':
            amt_median = np.median(amounts)
            amt_var = np.median(np.abs(amounts - amt_median)) # MAD
        else:
            amt_var = amounts.std()

        if amt_var > self.config.amount_variance_threshold:
            return False

        # 3. Check Date Stability
        gaps = dates.sort_values().diff().dt.days.dropna()
        if len(gaps) < 1: return False

        if self.config.stability_metric == 'mad':
            gap_median = np.median(gaps)
            gap_var = np.median(np.abs(gaps - gap_median))
        else:
            gap_var = gaps.std()

        return gap_var <= self.config.date_variance_threshold