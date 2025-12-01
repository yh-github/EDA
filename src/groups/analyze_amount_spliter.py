import sys
from pathlib import Path

import pandas as pd
import numpy as np
from collections import defaultdict


# --- 1. The Gap-Splitting Logic (from groups.by_amount) ---
def cluster_expenses_gap(numbers: list[float], ids: list[str], max_size: int, min_size: int):
    """
    Recursively splits a list of transactions by finding the WIDEST GAP in amounts.
    """
    if not numbers:
        return []

    # Sort based on numbers to find 1D gaps
    combined = sorted(zip(numbers, ids), key=lambda x: x[0])
    sorted_nums = [x[0] for x in combined]
    sorted_ids = [x[1] for x in combined]

    def recursive_split(segment_nums, segment_ids):
        # Base Case: Fits in window
        if len(segment_nums) <= max_size:
            return [(segment_nums, segment_ids)]

        # Find the Widest Gap to split on
        max_gap = -1
        split_index = -1
        mid_point = len(segment_nums) / 2

        # Constraints: Resulting chunks must be >= min_size
        start_search = min_size - 1
        end_search = len(segment_nums) - min_size

        # Fallback: If segment too small for min_size constraint, just search everywhere
        if start_search >= end_search:
            start_search = 0
            end_search = len(segment_nums) - 1

        for i in range(start_search, end_search):
            gap = segment_nums[i + 1] - segment_nums[i]
            # STRICTLY greater finds the first largest gap.
            # We use >= to break ties by proximity to center (balanced trees)
            if gap > max_gap:
                max_gap = gap
                split_index = i
            elif gap == max_gap:
                # Tie-breaker: Choose split closest to the middle (balance)
                if abs(i - mid_point) < abs(split_index - mid_point):
                    split_index = i

        # Safety: If no split found (e.g. all identical), force split at middle
        if split_index == -1:
            split_index = int(mid_point)

        # Execute Split
        return recursive_split(segment_nums[:split_index + 1], segment_ids[:split_index + 1]) + \
            recursive_split(segment_nums[split_index + 1:], segment_ids[split_index + 1:])

    return recursive_split(sorted_nums, sorted_ids)


# --- 2. The Analysis Harness ---
def analyze_splitting_impact(df_path:Path):
    print("Loading data...")
    # Load your dataset
    try:
        df = pd.read_csv(df_path)
    except FileNotFoundError:
        print("Dataset not found. Please ensure data/rec_data2.csv exists.")
        return

    # Filter for valid Ground Truth groups
    df_clean = df.dropna(subset=['patternId', 'amount'])
    # Filter out Noise (-1) and Nulls
    df_clean = df_clean[~df_clean['patternId'].isin(['-1', -1, 'None', 'nan'])]

    print(f"Analyzing {len(df_clean)} transactions with valid Pattern IDs...")

    # Config matching your model
    MAX_SEQ_LEN = 200
    MIN_SPLIT_SIZE = 32  # Don't create chunks smaller than this if possible

    total_groups = 0
    broken_groups = 0

    group_stats = []

    # Iterate Accounts
    for acc_id, acc_df in df_clean.groupby('accountId'):
        # We only care about accounts that ARE truncated/split
        if len(acc_df) <= MAX_SEQ_LEN:
            continue

        amounts = acc_df['amount'].tolist()
        # We track patternId to see if they get separated
        pids = acc_df['patternId'].tolist()

        # RUN THE METHOD
        clusters = cluster_expenses_gap(amounts, pids, MAX_SEQ_LEN, MIN_SPLIT_SIZE)

        # Map PatternID -> Set of Cluster Indices
        # If a PatternID appears in Cluster 0 and Cluster 1, it is BROKEN.
        pid_dist = defaultdict(set)
        for c_idx, (_, c_pids) in enumerate(clusters):
            for pid in c_pids:
                pid_dist[pid].add(c_idx)

        # Analyze Results per Pattern
        for pid in acc_df['patternId'].unique():
            total_groups += 1

            # Get Ground Truth Stats for this group
            grp_amts = acc_df[acc_df['patternId'] == pid]['amount'].values

            # Calculate the "Max Internal Gap" of the group
            # This tells us: "If we split by amount, is there a natural gap INSIDE this group?"
            sorted_grp = np.sort(grp_amts)
            max_internal_gap = 0.0
            if len(sorted_grp) > 1:
                max_internal_gap = np.max(np.diff(sorted_grp))

            is_broken = len(pid_dist[pid]) > 1
            if is_broken:
                broken_groups += 1

            group_stats.append({
                'broken': is_broken,
                'std_dev': np.std(grp_amts),
                'range': np.ptp(grp_amts),  # Max - Min
                'max_internal_gap': max_internal_gap,
                'size': len(grp_amts)
            })

    # --- 3. Report ---
    print("\n" + "=" * 60)
    print(f"SPLITTING METHOD: Max Gap (Amount Only)")
    print(f"Constraints: Max Size={MAX_SEQ_LEN}, Min Chunk={MIN_SPLIT_SIZE}")
    print("=" * 60)

    if total_groups == 0:
        print("No large accounts found with labeled patterns.")
        return

    print(f"Total Patterns Evaluated: {total_groups}")
    print(f"Broken Patterns:          {broken_groups}")
    print(f"Destruction Rate:         {broken_groups / total_groups:.2%}")

    stats_df = pd.DataFrame(group_stats)

    print("\n--- Diagnostic Stats (Helping you tune) ---")
    print("Compare the 'Internal Gaps' of Broken vs Intact groups.")
    print("If Broken groups have small gaps, it means we forced a split through dense data.")

    print("\nMean Values:")
    print(stats_df.groupby('broken')[['max_internal_gap', 'range', 'std_dev']].mean())

    print("\nPercentiles (95th):")
    print(stats_df.groupby('broken')[['max_internal_gap', 'range']].quantile(0.95))

    print("\n--- Recommendation ---")
    safe_gap = stats_df[stats_df['broken'] == False]['max_internal_gap'].quantile(0.95)
    print(f"Consider verifying if a split gap is < {safe_gap:.2f}. If so, fallback to Date Splitting.")


if __name__ == "__main__":
    analyze_splitting_impact(Path(sys.argv[1]))