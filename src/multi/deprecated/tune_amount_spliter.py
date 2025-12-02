import pandas as pd
from collections import defaultdict
from dataclasses import dataclass

from multi.config import MultiExpConfig
from multi.reload_utils import load_data_for_config


# --- 1. Hybrid Splitter ---
def cluster_expenses_hybrid(numbers: list[float], ids: list[str], max_size: int, min_size: int,
                            abs_thresh: float, rel_thresh: float):
    """
    Splits if gap > abs_thresh AND gap > rel_thresh (percentage).
    """
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

            # Relative Gap
            denom = abs(a) if abs(a) > 1e-9 else 1.0
            rel_diff = diff / denom

            # Hybrid Check: Must exceed BOTH thresholds to be considered
            # We use the relative difference as the "score" for maximization
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


# --- 2. Tuning Harness ---
@dataclass
class TuneResult:
    abs_t: float
    rel_t: float
    dest: float
    res: float


def run_hybrid_tuning(random_state:int|None=None, downsample:float|None=None):
    conf = MultiExpConfig()
    if random_state is not None:
        conf.random_state = random_state
    if downsample is not None:
        conf.downsample = downsample

    train_df, val_df, test_df = load_data_for_config(conf)

    df = train_df

    df = df.dropna(subset=['patternId', 'amount'])
    df = df[~df['patternId'].isin(['-1', -1, 'None', 'nan'])]

    MAX_SIZE = 200
    MIN_CHUNK = 32

    large_accounts = [g for _, g in df.groupby('accountId') if len(g) > MAX_SIZE]
    print(f"Benchmarking on {len(large_accounts)} large accounts.")

    # Hybrid Grid
    abs_grid = [5.0, 10.0, 20.0]
    rel_grid = [0.1, 0.2, 0.3, 0.5]  # 10%, 20%, 30%, 50%

    results = []

    print(f"\n{'Abs($)':<8} | {'Rel(%)':<8} | {'Destruction':<12} | {'Resolution':<12}")
    print("-" * 50)

    for abs_t in abs_grid:
        for rel_t in rel_grid:
            total_patterns = 0
            broken_patterns = 0
            total_txns = 0
            resolved_txns = 0

            for acc_df in large_accounts:
                chunks = cluster_expenses_hybrid(
                    acc_df['amount'].tolist(),
                    acc_df['patternId'].tolist(),
                    MAX_SIZE, MIN_CHUNK, abs_t, rel_t
                )

                # Metrics
                pid_map = defaultdict(set)
                for i, (c_nums, c_pids) in enumerate(chunks):
                    if len(c_nums) <= MAX_SIZE: resolved_txns += len(c_nums)
                    total_txns += len(c_nums)
                    for pid in c_pids: pid_map[pid].add(i)

                for pid in acc_df['patternId'].unique():
                    total_patterns += 1
                    if len(pid_map[pid]) > 1: broken_patterns += 1

            dest = broken_patterns / total_patterns if total_patterns else 0
            res = resolved_txns / total_txns if total_txns else 0

            print(f"{abs_t:<8} | {rel_t:<8.1%} | {dest:>11.2%} | {res:>11.2%}")
            results.append(TuneResult(abs_t, rel_t, dest, res))

    # Recommendation
    best = min(results, key=lambda x: (x.dest > 0.01, -x.res))  # prioritize dest < 1%, then max res
    print("\n🏆 Recommended Hybrid Config:")
    print(f"Abs > ${best.abs_t} AND Rel > {best.rel_t:.1%}")
    print(f"Destruction: {best.dest:.2%} | Resolution: {best.res:.2%}")


if __name__ == "__main__":
    run_hybrid_tuning()