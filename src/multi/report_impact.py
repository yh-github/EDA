import sys
import os
import pandas as pd
import numpy as np
from collections import defaultdict

# Ensure we can import from src
sys.path.append(os.path.abspath('src'))

from multi.config import MultiExpConfig
from multi.reload_utils import load_data_for_config


# --- 1. The Hybrid Splitter (Copied for stability) ---
def cluster_expenses_hybrid(numbers: list[float], ids: list[str], max_size: int, min_size: int,
                            abs_thresh: float, rel_thresh: float):
    if not numbers: return []

    combined = sorted(zip(numbers, ids), key=lambda x: x[0])
    sorted_nums = [x[0] for x in combined]
    sorted_ids = [x[1] for x in combined]

    def recursive_split(segment_nums, segment_ids):
        N = len(segment_nums)
        if N <= max_size: return [(segment_nums, segment_ids)]

        max_score = -1.0
        split_index = -1
        mid_point = N / 2

        start = min_size - 1
        end = N - min_size
        if start >= end: start, end = 0, N - 1

        for i in range(start, end):
            a = segment_nums[i]
            b = segment_nums[i + 1]
            diff = b - a

            denom = abs(a) if abs(a) > 1e-9 else 1.0
            rel_diff = diff / denom

            if diff > abs_thresh and rel_diff > rel_thresh:
                if rel_diff > max_score:
                    max_score = rel_diff
                    split_index = i
                elif rel_diff == max_score:
                    if abs(i - mid_point) < abs(split_index - mid_point):
                        split_index = i

        if split_index == -1:
            return [(segment_nums, segment_ids)]

        return recursive_split(segment_nums[:split_index + 1], segment_ids[:split_index + 1]) + \
            recursive_split(segment_nums[split_index + 1:], segment_ids[split_index + 1:])

    return recursive_split(sorted_nums, sorted_ids)


# --- 2. Impact Analysis ---
def analyze_impact():
    print("Loading Data (Train + Val)...")
    try:
        conf = MultiExpConfig()
        train_df, val_df, test_df = load_data_for_config(conf)
        df = pd.concat([train_df, val_df], ignore_index=True)
        print(f"Loaded {len(df)} transactions.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Filter Valid Ground Truth
    df = df.dropna(subset=['patternId', 'amount'])
    df = df[~df['patternId'].isin(['-1', -1, 'None', 'nan'])]

    # --- CONFIGURATION (Use your Winning Params) ---
    MAX_SIZE = 200
    MIN_CHUNK = 32
    ABS_THRESH = 5.0
    REL_THRESH = 0.30

    print(f"\nConfiguration: Window={MAX_SIZE}, Abs>${ABS_THRESH}, Rel>{REL_THRESH:.0%}")

    # 1. Account Sizing
    acc_counts = df.groupby('accountId').size()
    large_acc_ids = acc_counts[acc_counts > MAX_SIZE].index

    large_df = df[df['accountId'].isin(large_acc_ids)]

    num_total_acc = df['accountId'].nunique()
    num_large_acc = len(large_acc_ids)

    num_total_txns = len(df)
    num_large_txns = len(large_df)

    print(f"\n--- Account Scope ---")
    print(f"Total Accounts: {num_total_acc}")
    print(f"Large Accounts: {num_large_acc} ({num_large_acc / num_total_acc:.2%})")
    print(f"Large Txns:     {num_large_txns} ({num_large_txns / num_total_txns:.2%})")

    if num_large_acc == 0:
        print("No accounts larger than window size. Splitting not required!")
        return

    # 2. Run Splitting
    print(f"\nRunning analysis on {num_large_acc} accounts...")

    broken_stats = []  # Store stats for every group (broken or not)

    affected_accounts = 0
    affected_txns = 0
    affected_groups = 0
    total_large_groups = large_df['patternId'].nunique()

    for acc_id, acc_df in large_df.groupby('accountId'):
        amts = acc_df['amount'].tolist()
        pids = acc_df['patternId'].tolist()

        chunks = cluster_expenses_hybrid(amts, pids, MAX_SIZE, MIN_CHUNK, ABS_THRESH, REL_THRESH)

        # Check Integrity
        pid_map = defaultdict(set)
        for i, (_, c_pids) in enumerate(chunks):
            for pid in c_pids:
                pid_map[pid].add(i)

        is_acc_broken = False

        for pid in acc_df['patternId'].unique():
            is_broken = len(pid_map[pid]) > 1

            # Collect Stats for this group
            grp_data = acc_df[acc_df['patternId'] == pid]['amount'].values
            if len(grp_data) > 1:
                gaps = np.diff(np.sort(grp_data))
                max_gap = np.max(gaps)
            else:
                max_gap = 0.0

            broken_stats.append({
                'broken': is_broken,
                'size': len(grp_data),
                'range': np.ptp(grp_data),
                'max_gap': max_gap,
                'std': np.std(grp_data)
            })

            if is_broken:
                affected_groups += 1
                affected_txns += len(grp_data)
                is_acc_broken = True

        if is_acc_broken:
            affected_accounts += 1

    # 3. Report
    print("\n" + "=" * 60)
    print("  IMPACT REPORT")
    print("=" * 60)

    print("1. Account Impact (How many users get a degraded experience?)")
    print(f"   Large Accounts Broken: {affected_accounts} / {num_large_acc} ({affected_accounts / num_large_acc:.2%})")
    print(f"   Global Accounts Broken: {affected_accounts} / {num_total_acc} ({affected_accounts / num_total_acc:.2%})")

    print("\n2. Group Impact (How many patterns are split?)")
    print(
        f"   Large Groups Broken: {affected_groups} / {total_large_groups} ({affected_groups / total_large_groups:.2%})")

    print("\n3. Transaction Impact (How much data loses context?)")
    print(f"   Large Txns Affected: {affected_txns} / {num_large_txns} ({affected_txns / num_large_txns:.2%})")

    # 4. Diagnostics
    print("\n" + "-" * 60)
    print("  WHY DID THEY BREAK?")
    print("-" * 60)
    stats_df = pd.DataFrame(broken_stats)

    print("Comparison of Group Internals (Broken vs Intact):")
    print(stats_df.groupby('broken')[['max_gap', 'range', 'std']].mean().to_string())

    print("\nInterpretation:")
    print(" - High 'max_gap' in Broken groups = Good! We split sparse groups (e.g. distinct clusters).")
    print(" - Low 'max_gap' in Broken groups = Bad. We forced a split in dense data.")


if __name__ == "__main__":
    analyze_impact()