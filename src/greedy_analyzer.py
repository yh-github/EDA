import logging
import numpy as np
import pandas as pd
import torch
from datetime import timedelta
from typing import List, Literal
from difflib import SequenceMatcher
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
                 amount_tol_pct: float = 0.05,
                 metric: Literal["cosine", "lexical"] = "cosine"):

        self.filter_config = filter_config
        self.fields = field_config
        self.sim_threshold = sim_threshold
        self.amount_tol_abs = amount_tol_abs
        self.amount_tol_pct = amount_tol_pct
        self.metric = metric

    def analyze_account(self, account_df: pd.DataFrame, embeddings: np.ndarray | None) -> List[RecurringGroupResult]:
        if account_df.empty:
            return []

        # 1. Setup
        df_reset = account_df.reset_index(drop=True)
        ids = df_reset[self.fields.trId].values
        dates = pd.to_datetime(df_reset[self.fields.date])
        amounts = df_reset[self.fields.amount].values
        texts = df_reset[self.fields.text].values

        # 2. Pre-calculate Similarity Matrix (Only for Cosine)
        sim_matrix = None
        if self.metric == "cosine":
            if embeddings is None or len(embeddings) == 0:
                raise ValueError("Embeddings required for cosine metric")
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

            # --- A. Similarity Filter ---
            candidates = []

            if self.metric == "cosine":
                # Vectorized lookup
                sim_scores = sim_matrix[i].numpy()
                # Find indices where sim > threshold
                candidates_indices = np.where(sim_scores > self.sim_threshold)[0]
                # Exclude already assigned
                candidates = candidates_indices[~assigned_mask[candidates_indices]]

            elif self.metric == "lexical":
                # Iterative String Matching
                anchor_text = str(texts[i])
                potential_matches = []

                for j in range(n_samples):
                    if assigned_mask[j]: continue
                    # Optimization: Skip self in inner loop if needed,
                    # but typically self-similarity is 1.0, so we keep it.

                    # Calculate similarity
                    # Note: This is O(N^2) string comparisons per account.
                    # For N < 1000, this is acceptable (~0.5s).
                    text_j = str(texts[j])
                    score = SequenceMatcher(None, anchor_text, text_j).ratio()

                    if score > self.sim_threshold:
                        potential_matches.append(j)

                candidates = np.array(potential_matches)

            if len(candidates) < self.filter_config.min_txns_for_period:
                continue

            # --- B. Amount Filter ---
            anchor_amt = amounts[i]
            cand_amounts = amounts[candidates]
            abs_diff = np.abs(cand_amounts - anchor_amt)

            amt_mask = (abs_diff <= self.amount_tol_abs) | \
                       (abs_diff <= (np.abs(anchor_amt) * self.amount_tol_pct))

            valid_indices = candidates[amt_mask]

            if len(valid_indices) < self.filter_config.min_txns_for_period:
                continue

            # --- C. Date/Cycle Filter ---
            group_dates = dates.iloc[valid_indices].sort_values()
            gaps_series = group_dates.diff().dt.days.dropna()

            if len(gaps_series) < (self.filter_config.min_txns_for_period - 1):
                continue

            gaps = gaps_series.values

            # Robust vs Standard Variance
            if self.filter_config.stability_metric == 'mad':
                gap_median = float(np.median(gaps))
                gap_var = float(np.median(np.abs(gaps - gap_median)))
            else:
                gap_var = float(gaps.std())
                if np.isnan(gap_var): gap_var = 0.0

            if gap_var > self.filter_config.date_variance_threshold:
                continue

            # --- D. Group Found ---
            assigned_mask[valid_indices] = True
            cluster_id_counter += 1

            # Stats
            median_gap = float(np.median(gaps))
            amt_vals = amounts[valid_indices]
            amt_mean = float(np.mean(amt_vals))

            result = RecurringGroupResult(
                cluster_id=cluster_id_counter,
                status=GroupStabilityStatus.STABLE,
                transaction_ids=ids[valid_indices].tolist(),
                cycle_days=median_gap,
                predicted_amount=amt_mean,
                next_predicted_date=group_dates.iloc[-1] + timedelta(days=median_gap),
                amount_mean=amt_mean,
                amount_std=float(np.std(amt_vals)),
                date_delta_mean_days=float(np.mean(gaps)),
                date_delta_std_days=float(gaps.std()) if len(gaps) > 1 else 0.0
            )
            results.append(result)

        return results