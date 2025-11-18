import logging
import numpy as np
import pandas as pd
import torch
from datetime import timedelta
from typing import List, Optional

# Reuse existing data structures for compatibility
from config import FieldConfig, FilterConfig
from group_analyzer import RecurringGroupResult, GroupStabilityStatus

logger = logging.getLogger(__name__)


class GreedyGroupAnalyzer:
    """
    Identifies recurring transaction groups using a 'Greedy Anchor' approach.

    Instead of global clustering (DBSCAN), this algorithm:
    1. Treats every transaction as a potential 'Anchor'.
    2. Finds all other transactions with high Semantic Similarity (1-vs-All).
    3. Filters those neighbors strictly by Amount (must be very close).
    4. Filters by Time (must form a regular cycle).
    5. Extracts the group if it passes all checks, marking items as 'assigned'.
    """

    def __init__(self, filter_config: FilterConfig, field_config: FieldConfig):
        self.filter_config = filter_config
        self.fields = field_config

        # --- Tunable Thresholds (Override defaults if needed) ---
        # Minimum cosine similarity to consider a transaction a "candidate" match
        self.sim_threshold = 0.90

        # Amount tolerance: Match if within $2.00 OR within 5% (whichever is larger)
        self.amount_tol_abs = 2.00
        self.amount_tol_pct = 0.05

    def analyze_account(
            self,
            account_df: pd.DataFrame,
            embeddings: np.ndarray
    ) -> List[RecurringGroupResult]:
        """
        Analyzes a single account to find stable recurring groups.
        """
        if account_df.empty or len(embeddings) == 0:
            return []

        # 1. Setup Data Vectors for fast access
        # Reset index to ensure 0..N alignment with embeddings
        df_reset = account_df.reset_index(drop=True)

        ids = df_reset[self.fields.trId].values
        dates = pd.to_datetime(df_reset[self.fields.date])
        amounts = df_reset[self.fields.amount].values
        # We don't strictly need text here as embeddings cover semantics

        # 2. Compute Similarity Matrix (N x N) using PyTorch
        # Normalize vectors first so dot_product == cosine_similarity
        emb_tensor = torch.from_numpy(embeddings).float()
        emb_tensor = torch.nn.functional.normalize(emb_tensor, p=2, dim=1)

        # Matrix Multiplication: (N x 768) @ (768 x N) -> (N x N)
        # This gives us the similarity of every transaction to every other transaction
        sim_matrix = torch.mm(emb_tensor, emb_tensor.t())

        # 3. Greedy Iteration
        n_samples = len(df_reset)
        assigned_mask = np.zeros(n_samples, dtype=bool)
        results: List[RecurringGroupResult] = []
        cluster_id_counter = 0

        # Iterate through every transaction to see if it anchors a group
        for i in range(n_samples):
            # Skip if already claimed by a previous strong group
            if assigned_mask[i]:
                continue

            # --- A. Semantic Filter ---
            # Find all indices where similarity > threshold
            # .numpy() is cheap here for a single row
            sim_scores = sim_matrix[i].numpy()
            candidates = np.where(sim_scores > self.sim_threshold)[0]

            # Remove candidates that are already assigned to other groups
            candidates = candidates[~assigned_mask[candidates]]

            # Need at least 'min_txns' to form a group (default 3)
            if len(candidates) < self.filter_config.min_txns_for_period:
                continue

            # --- B. Amount Filter ---
            # Check strict amount proximity to the Anchor (index i)
            anchor_amt = amounts[i]
            cand_amounts = amounts[candidates]

            abs_diff = np.abs(cand_amounts - anchor_amt)
            # Allow small absolute variance OR small percentage variance
            amt_mask = (abs_diff <= self.amount_tol_abs) | \
                       (abs_diff <= (np.abs(anchor_amt) * self.amount_tol_pct))

            valid_indices = candidates[amt_mask]

            if len(valid_indices) < self.filter_config.min_txns_for_period:
                continue

            # --- C. Date/Cycle Filter ---
            # We need to check if these transactions occur on a regular schedule.
            group_dates = dates.iloc[valid_indices].sort_values()

            # Calculate gaps in days
            gaps = group_dates.diff().dt.days.dropna()

            if len(gaps) == 0:
                continue

            median_gap = gaps.median()
            gap_std = gaps.std()

            # If only 2 transactions (1 gap), std is NaN.
            # If config requires 3 txns, we have at least 2 gaps, so std is valid.
            # If we allow 2 txns, we treat std as 0.0 (perfect stability).
            if np.isnan(gap_std):
                gap_std = 0.0

            # Check against stability threshold from config
            if gap_std > self.filter_config.date_std_threshold:
                # Too jittery to be a subscription (e.g. random coffee trips)
                continue

            # --- D. Group Found! ---
            # Mark these indices as assigned so they aren't reused
            assigned_mask[valid_indices] = True
            cluster_id_counter += 1

            # Calculate Forecast stats
            last_date = group_dates.iloc[-1]
            predicted_date = last_date + timedelta(days=median_gap)

            # Create the Result Object
            result = RecurringGroupResult(
                cluster_id=cluster_id_counter,
                status=GroupStabilityStatus.STABLE,
                transaction_ids=ids[valid_indices].tolist(),
                cycle_days=float(median_gap),
                predicted_amount=float(np.mean(amounts[valid_indices])),
                next_predicted_date=predicted_date,
                amount_mean=float(np.mean(amounts[valid_indices])),
                amount_std=float(np.std(amounts[valid_indices])),
                date_delta_mean_days=float(np.mean(gaps)),
                date_delta_std_days=float(gap_std)
            )
            results.append(result)

        return results