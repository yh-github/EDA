import pandas as pd


def calculate_group_metrics(val_results_path):
    print(f"Loading {val_results_path}...")
    df = pd.read_csv(val_results_path)

    # Ensure booleans
    df['isRecurring'] = df['isRecurring'].fillna(False).astype(bool)

    # --- Group Precision ---
    # Definition: What % of detected groups are actually recurring?
    # A detected group is "True" if > 50% of its transactions are recurring.

    predicted_groups = df[df['recurring_group_id'] != -1].groupby('recurring_group_id')

    if len(predicted_groups) == 0:
        print("No groups detected.")
        return

    true_positive_groups = 0
    total_predicted_groups = len(predicted_groups)

    for g_id, group in predicted_groups:
        if group['isRecurring'].mean() > 0.5:
            true_positive_groups += 1

    group_precision = true_positive_groups / total_predicted_groups if total_predicted_groups > 0 else 0

    # --- Group Recall ---
    # Definition: What % of ground truth patterns were found?
    # A GT pattern is "Found" if > 50% of its transactions are detected (assigned to any group).

    # Filter for recurring transactions only to identify GT patterns
    recurring_txns = df[df['isRecurring'] == True]

    if len(recurring_txns) == 0:
        print("No recurring transactions in ground truth.")
        return

    # Group by patternId
    # Note: patternId might be null for some recurring txns if not assigned a pattern?
    # Let's assume patternId is the identifier.
    if 'patternId' not in df.columns:
        print("patternId column missing. Cannot calculate Group Recall.")
        return

    gt_groups = recurring_txns.groupby('patternId')
    total_gt_groups = len(gt_groups)
    detected_gt_groups = 0

    for p_id, group in gt_groups:
        # Check how many transactions in this GT group were detected (id != -1)
        n_detected = (group['recurring_group_id'] != -1).sum()
        detection_rate = n_detected / len(group)

        if detection_rate > 0.5:
            detected_gt_groups += 1

    group_recall = detected_gt_groups / total_gt_groups if total_gt_groups > 0 else 0

    group_f1 = 2 * (group_precision * group_recall) / (group_precision + group_recall) if (
                                                                                                      group_precision + group_recall) > 0 else 0

    print("\n--- Group-Level Metrics ---")
    print(f"Total Predicted Groups: {total_predicted_groups}")
    print(f"True Positive Groups: {true_positive_groups}")
    print(f"Group Precision: {group_precision:.4f}")
    print("-" * 20)
    print(f"Total Ground Truth Groups: {total_gt_groups}")
    print(f"Detected GT Groups: {detected_gt_groups}")
    print(f"Group Recall: {group_recall:.4f}")
    print("-" * 20)
    print(f"Group F1 Score: {group_f1:.4f}")


if __name__ == "__main__":
    calculate_group_metrics('val_results.csv')
