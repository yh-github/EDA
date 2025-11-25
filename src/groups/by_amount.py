def cluster_expenses(numbers: list[float], max_size: int, min_size: int) -> list[list[float]]:
    """
    Clusters a list of numbers such that:
    1. No cluster has more items than max_size.
    2. If split, resulting clusters (and siblings) prefer to be >= min_size.

    Splits are made at the widest gaps within the valid range to preserve data separation.
    """
    print(f"{max_size=} {min_size=}")
    if not numbers:
        return []

    # 1. Sort the data (Crucial for 1D clustering)
    sorted_nums = sorted(numbers)

    def recursive_split(segment: list[float]) -> list[list[float]]:
        # Base Case: If the segment is empty, return
        if not segment:
            return []

        # Base Case: Single element is valid unless strictly filtered later,
        # but recursion usually stops before this if constraints are met.
        if len(segment) <= 1:
            return [segment]

        # Check the SIZE constraint (Count of items)
        if len(segment) <= max_size:
            return [segment]

        # If len > MAX_SIZE, we MUST split.
        max_gap = -1
        split_index = -1
        mid_point = len(segment) / 2

        # 2. Determine valid split range to respect MIN_SIZE
        # We need: i+1 >= min_size AND len - (i+1) >= min_size
        start_search = min_size - 1
        end_search = len(segment) - min_size

        # Edge Case: If segment is too small to be split into two min_sizes
        # (e.g. len=5, min=3 -> needs 6 items), we cannot satisfy min_size.
        # We fallback to searching the whole range to satisfy MAX_SIZE (harder constraint).
        if start_search >= end_search:
            start_search = 0
            end_search = len(segment) - 1

        # 3. Find widest gap within the valid split range
        for i in range(start_search, end_search):
            gap = segment[i + 1] - segment[i]

            # Update if we find a strictly larger gap
            if gap > max_gap:
                max_gap = gap
                split_index = i
            # Tie-breaker: choose split closest to center
            elif gap == max_gap:
                current_dist = abs(i - mid_point)
                best_dist = abs(split_index - mid_point)
                if current_dist < best_dist:
                    split_index = i

        # Perform the split
        left_segment = segment[:split_index + 1]
        right_segment = segment[split_index + 1:]

        # Recurse on both halves
        return recursive_split(left_segment) + recursive_split(right_segment)

    return recursive_split(sorted_nums)


def print_analysis(clusters: list[list[float]]):
    """
    Helper to print clusters with gap analysis.
    """
    if not clusters:
        print("No clusters found.")
        return

    print(f"{'Cluster':<15} {'Size':<6} {'Min Internal Gap':<18} {'Gap to Next':<12} {'Data'}")
    print("-" * 80)

    for i, cluster in enumerate(clusters):
        # 1. Calculate Internal Min Gap (Density)
        min_internal = "N/A"
        if len(cluster) > 1:
            gaps = [cluster[j + 1] - cluster[j] for j in range(len(cluster) - 1)]
            min_internal = f"{min(gaps):.2f}"

        # 2. Calculate Gap to Next Cluster (Separation)
        gap_next = "End"
        if i < len(clusters) - 1:
            next_cluster_start = clusters[i + 1][0]
            current_cluster_end = cluster[-1]
            gap_next = f"{next_cluster_start - current_cluster_end:.2f}"

        # 3. Format Data
        data_str = str(cluster)
        if len(data_str) > 30:
            data_str = data_str[:27] + "..."

        print(f"Group {i + 1:<9} {len(cluster):<6} {min_internal:<18} {gap_next:<12} {data_str}")


# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: Clear separation
    expenses = [10, 12, 15, 100, 102, 105, 500, 501]

    clusters = cluster_expenses(expenses, max_size=3, min_size=2)
    print_analysis(clusters)

    # Example 2: Continuous data
    expenses_dense = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    clusters_dense = cluster_expenses(expenses_dense,  max_size=5, min_size=3)
    print_analysis(clusters_dense)

    # Example 3: Repeating numbers
    expenses_repeats = [9.9, 10, 10, 10, 10, 10, 50, 50, 50]

    clusters_repeats = cluster_expenses(expenses_repeats, max_size=5, min_size=2)
    print_analysis(clusters_repeats)

    expenses_min = [1, 2, 10, 11, 12, 13, 20, 21]

    clusters_min = cluster_expenses(expenses_min, max_size=5, min_size=3)
    print_analysis(clusters_min)
    print_analysis(clusters_min)