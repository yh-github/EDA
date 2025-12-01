import pandas as pd
import numpy as np
from collections import defaultdict
from dataclasses import dataclass


# --- 1. Configurable Splitter ---
def cluster_expenses_tune(numbers: list[float], ids: list[str], max_size: int, min_size: int,
                          metric: str = 'absolute', threshold: float = 0.0):
    """
    Splits data recursively. Stops if the best gap found is < threshold.
    metric: 'absolute' (b - a) or 'relative' ((b - a) / a)
    """
    if not numbers:
        return []

    # Sort
    combined = sorted(zip(numbers, ids), key=lambda x: x[0])
    sorted_nums = [x[0] for x in combined]
    sorted_ids = [x[1] for x in combined]

    def recursive_split(segment_nums, segment_ids):
        N = len(segment_nums)
        # Success: Fits in window
        if N <= max_size:
            return [(segment_nums, segment_ids)]

        # Find Widest Gap
        max_gap = -1.0
        split_index = -1
        mid_point = N / 2

        start_search = min_size - 1
        end_search = N - min_size

        # Fallback for small chunks
        if start_search >= end_search:
            start_search = 0
            end_search = N - 1

        for i in range(start_search, end_search):
            a = segment_nums[i]
            b = segment_nums[i + 1]

            diff = b - a

            if metric == 'absolute':
                gap_val = diff
            else:  # relative
                # Avoid div by zero. Use average or min.
                # (b-a)/a is standard relative growth.
                denom = abs(a) if abs(a) > 1e-9 else 1.0
                gap_val = diff / denom

            if gap_val > max_gap:
                max_gap = gap_val
                split_index = i
            elif gap_val == max_gap:
                if abs(i - mid_point) < abs(split_index - mid_point):
                    split_index = i

        # --- THRESHOLD CHECK ---
        # If the BEST gap we found is trivial (e.g. < 1 cent or < 1%),
        # we refuse to split by amount.
        if split_index == -1 or max_gap <= threshold:
            # We return the whole chunk.
            # The caller will see it's > max_size and count it as "Unresolved" (fallback to time)
            return [(segment_nums, segment_ids)]

        # Execute Split
        return recursive_split(segment_nums[:split_index + 1], segment_ids[:split_index + 1]) + \
            recursive_split(segment_nums[split_index + 1:], segment_ids[split_index + 1:])

    return recursive_split(sorted_nums, sorted_ids)


# --- 2. Tuning Harness ---
@dataclass
class TuneResult:
    method: str
    threshold: float
    destruction_rate: float  # % of Patterns broken
    resolution_rate: float  # % of Transactions successfully fit into chunks <= MAX_SIZE
    broken_mean_amt: float  # Avg amount of broken groups (Diagnostic)


def run_tuning():
    print("Loading data...")
    try:
        df = pd.read_csv('data/rec_data2.csv')
    except:
        print("Data not found.")
        return

    # Filter
    df = df.dropna(subset=['patternId', 'amount'])
    df = df[~df['patternId'].isin(['-1', -1, 'None', 'nan'])]

    # 1. Identify Accounts that NEED splitting
    # We only benchmark on accounts > window size
    MAX_SIZE = 200
    MIN_CHUNK = 32

    large_accounts = []
    for acc_id, group in df.groupby('accountId'):
        if len(group) > MAX_SIZE:
            large_accounts.append(group)

    print(f"Benchmarking on {len(large_accounts)} large accounts ({sum(len(x) for x in large_accounts)} txns).")
    print(f"Target Window: {MAX_SIZE}")

    # Define Grid
    # Absolute: 0.01 (1 cent) to 50.0 (50 dollars)
    # Relative: 0.001 (0.1%) to 0.5 (50%)
    grid = [
        ('absolute', 0.01), ('absolute', 1.0), ('absolute', 5.0), ('absolute', 10.0), ('absolute', 50.0),
        ('relative', 0.001), ('relative', 0.01), ('relative', 0.05), ('relative', 0.1), ('relative', 0.2)
    ]

    results = []

    print(f"\n{'Method':<10} | {'Thresh':<8} | {'Destruction':<12} | {'Resolution':<12} | {'Avg Broken Amt':<15}")
    print("-" * 75)

    for method, thresh in grid:
        total_patterns = 0
        broken_patterns = 0

        total_txns = 0
        resolved_txns = 0

        broken_amts = []

        for acc_df in large_accounts:
            amts = acc_df['amount'].tolist()
            pids = acc_df['patternId'].tolist()

            # Run Splitter
            chunks = cluster_expenses_tune(amts, pids, MAX_SIZE, MIN_CHUNK, method, thresh)

            # 1. Calc Resolution (Efficiency)
            # A chunk is resolved if len <= MAX_SIZE
            for c_nums, c_pids in chunks:
                if len(c_nums) <= MAX_SIZE:
                    resolved_txns += len(c_nums)
                total_txns += len(c_nums)

            # 2. Calc Destruction (Safety)
            pid_map = defaultdict(set)
            for i, (c_nums, c_pids) in enumerate(chunks):
                for pid in c_pids:
                    pid_map[pid].add(i)

            acc_pids = acc_df['patternId'].unique()
            for pid in acc_pids:
                total_patterns += 1
                if len(pid_map[pid]) > 1:
                    broken_patterns += 1
                    # Log mean amount of this broken group
                    grp_mean = acc_df[acc_df['patternId'] == pid]['amount'].mean()
                    broken_amts.append(grp_mean)

        dest_rate = broken_patterns / total_patterns if total_patterns else 0
        res_rate = resolved_txns / total_txns if total_txns else 0
        mean_broken = np.mean(broken_amts) if broken_amts else 0.0

        print(f"{method:<10} | {thresh:<8} | {dest_rate:>11.2%} | {res_rate:>11.2%} | {mean_broken:>14.2f}")

        results.append(TuneResult(method, thresh, dest_rate, res_rate, mean_broken))

    # --- Best Pick ---
    # We want max Resolution where Destruction < 1%
    valid_configs = [r for r in results if r.destruction_rate < 0.01]
    if valid_configs:
        best = max(valid_configs, key=lambda x: x.resolution_rate)
        print("\n🏆 Recommended Config:")
        print(f"Method: {best.method}, Threshold: {best.threshold}")
        print(f"Destruction: {best.destruction_rate:.2%}, Resolution: {best.resolution_rate:.2%}")

        if best.resolution_rate > 0.98:
            print("🚀 Excellent! Amount splitting handles >98% of cases safely. You can reduce Window Size!")
    else:
        print("\nNo config found with < 1% destruction. Consider relaxing goal or using best available.")


if __name__ == "__main__":
    run_tuning()