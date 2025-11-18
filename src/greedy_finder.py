import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from config import FieldConfig


@dataclass
class PatternGroup:
    main_text: str
    amount: float
    frequency_days: float
    transaction_ids: list[str]


class GreedyPatternFinder:
    def __init__(self, field_config: FieldConfig):
        self.fc = field_config

        # Tunable Thresholds
        self.sim_threshold = 0.85  # Minimum cosine similarity to be "Candidate"
        self.amount_tol_pct = 0.05  # Amount must be within 5%
        self.amount_tol_abs = 2.00  # OR within $2.00 (for small charges)
        self.date_jitter_days = 2.0  # Max allowed jitter in the cycle

    def find_patterns(self, df: pd.DataFrame, embeddings: np.ndarray) -> list[PatternGroup]:
        """
        Iteratively finds recurring groups by treating each transaction as an anchor.
        """
        if df.empty:
            return []

        # 1. Setup Data
        ids = df[self.fc.trId].values
        dates = pd.to_datetime(df[self.fc.date])
        amounts = df[self.fc.amount].values
        texts = df[self.fc.text].values

        # Normalize embeddings for Cosine Similarity
        # (A . B) / (|A|*|B|)  => If normalized, just (A . B)
        emb_tensor = torch.tensor(embeddings)
        emb_tensor = torch.nn.functional.normalize(emb_tensor, p=2, dim=1)

        # 2. Calculate Similarity Matrix (N x N)
        # For a single account (usually < 1000 txns), this is extremely fast
        sim_matrix = torch.mm(emb_tensor, emb_tensor.t()).numpy()

        # Mask to keep track of who has been assigned to a group
        assigned_mask = np.zeros(len(df), dtype=bool)
        found_groups = []

        # 3. Greedy Loop
        # Sort candidates by count (density) could be an optimization,
        # but linear scan is fine for now.
        for i in range(len(df)):
            if assigned_mask[i]:
                continue

            # --- Step A: Semantic Filter ---
            # Find all indices where similarity > threshold
            candidates = np.where(sim_matrix[i] > self.sim_threshold)[0]

            # Filter out those already assigned
            candidates = candidates[~assigned_mask[candidates]]

            if len(candidates) < 3:  # Need at least 3 for a pattern
                continue

            # --- Step B: Amount Filter ---
            # Check strictly against the Anchor's amount
            anchor_amt = amounts[i]

            # Create Boolean Mask for Amount
            # (abs(diff) < $2.00)  OR  (pct_diff < 5%)
            diffs = np.abs(amounts[candidates] - anchor_amt)
            amt_mask = (diffs < self.amount_tol_abs) | (diffs < (abs(anchor_amt) * self.amount_tol_pct))

            valid_candidates = candidates[amt_mask]

            if len(valid_candidates) < 3:
                continue

            # --- Step C: Date/Cycle Filter ---
            # Extract dates for these candidates
            candidate_dates = dates.iloc[valid_candidates].sort_values()

            # Calculate gaps (diff in days)
            gaps = candidate_dates.diff().dt.days.dropna()

            if len(gaps) < 2:
                continue

            median_gap = gaps.median()
            gap_std = gaps.std()

            # Simple Check: Is the jitter low?
            # Or more robust: Do most gaps align with the median?
            is_stable = gap_std < self.date_jitter_days

            if is_stable:
                # --- FOUND A GROUP! ---
                # Mark these as assigned
                assigned_mask[valid_candidates] = True

                group = PatternGroup(
                    main_text=texts[i],
                    amount=anchor_amt,
                    frequency_days=median_gap,
                    transaction_ids=ids[valid_candidates].tolist()
                )
                found_groups.append(group)

        return found_groups