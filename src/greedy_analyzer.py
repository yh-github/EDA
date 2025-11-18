import logging
import numpy as np
import pandas as pd
import torch
from datetime import timedelta
from typing import List
from config import FieldConfig, FilterConfig
from group_analyzer import RecurringGroupResult, GroupStabilityStatus

logger = logging.getLogger(__name__)


class GreedyGroupAnalyzer:
    def __init__(self,
                 filter_config: FilterConfig,
                 field_config: FieldConfig,
                 # Tunable parameters
                 sim_threshold: float = 0.90,
                 amount_tol_abs: float = 2.00,
                 amount_tol_pct: float = 0.05):

        self.filter_config = filter_config
        self.fields = field_config
        self.sim_threshold = sim_threshold
        self.amount_tol_abs = amount_tol_abs
        self.amount_tol_pct = amount_tol_pct

    def analyze_account(self, account_df: pd.DataFrame, embeddings: np.ndarray) -> List[RecurringGroupResult]:
        if account_df.empty or len(embeddings) == 0:
            return []

        # 1. Setup
        df_reset = account_df.reset_index(drop=True)
        ids = df_reset[self.fields.trId].values
        dates = pd.to_datetime(df_reset[self.fields.date])
        amounts = df_reset[self.fields.amount].values

        # 2. Similarity
        emb_tensor = torch.from_numpy(embeddings).float()
        emb_tensor = torch.nn.functional.normalize(emb_tensor, p=2, dim=1)
        sim_matrix = torch.mm(emb_tensor, emb_tensor.t())

        # 3. Greedy Iteration
        n_samples = len(df_reset)
        assigned_mask = np.zeros(n_samples, dtype=bool)
        results: List[RecurringGroupResult] = []
        cluster_id_counter = 0

        for i in range(n_samples):
            if assigned_mask[i]: continue

            # --- A. Semantic Filter ---
            sim_scores = sim_matrix[i].numpy()
            candidates = np.where(sim_scores > self.sim_threshold)[0]
            candidates = candidates[~assigned_mask[candidates]]

            if len(candidates) < self.filter_config.min_txns_for_period:
                continue

            # --- B. Amount Filter ---
            anchor_amt = amounts[i]
            cand_amounts = amounts[candidates]
            abs_diff = np.abs(cand_amounts - anchor_amt)

            # Allow small absolute OR small percentage variance
            amt_mask = (abs_diff <= self.amount_tol_abs) | \
                       (abs_diff <= (np.abs(anchor_amt) * self.amount_tol_pct))

            valid_indices = candidates[amt_mask]

            if len(valid_indices) < self.filter_config.min_txns_for_period:
                continue

            # --- C. Date/Cycle Filter ---
            group_dates = dates.iloc[valid_indices].sort_values()
            gaps = group_dates.diff().dt.days.dropna()

            if len(gaps) == 0: continue

            gap_std = gaps.std()
            if np.isnan(gap_std): gap_std = 0.0

            if gap_std > self.filter_config.date_std_threshold:
                continue

            # --- D. Group Found ---
            assigned_mask[valid_indices] = True
            cluster_id_counter += 1
            median_gap = gaps.median()

            result = RecurringGroupResult(
                cluster_id=cluster_id_counter,
                status=GroupStabilityStatus.STABLE,
                transaction_ids=ids[valid_indices].tolist(),
                cycle_days=float(median_gap),
                predicted_amount=float(np.mean(amounts[valid_indices])),
                next_predicted_date=group_dates.iloc[-1] + timedelta(days=median_gap),
                # Just basic stats for result
                amount_mean=float(np.mean(amounts[valid_indices])),
                amount_std=float(np.std(amounts[valid_indices])),
                date_delta_mean_days=float(np.mean(gaps)),
                date_delta_std_days=float(gap_std)
            )
            results.append(result)

        return results