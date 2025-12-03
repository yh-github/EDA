import pandas as pd
import numpy as np


def analyze_results(df, detected_groups):
    """
    Analyzes the performance of the detection.
    """
    print("Analyzing results...")

    # Basic stats
    n_transactions = len(df)
    n_recurring = len(df[df['recurring_group_id'] != -1])
    print(f"Total transactions: {n_transactions}")
    print(f"Detected recurring: {n_recurring} ({n_recurring / n_transactions:.2%})")

    # Group stats
    print("\nGroup Statistics:")
    print(detected_groups.describe())

    # Check against ground truth if available
    if 'isRecurring' in df.columns:
        print("\nGround Truth Comparison:")

        # Calculate precision/recall
        has_pattern = df['isRecurring'].fillna(False).astype(bool)
        is_detected = df['recurring_group_id'] != -1

        tp = (has_pattern & is_detected).sum()
        fp = (~has_pattern & is_detected).sum()
        fn = (has_pattern & ~is_detected).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")

        # Identify False Positives and False Negatives
        fp_mask = ~has_pattern & is_detected
        fn_mask = has_pattern & ~is_detected

        fp_examples = df[fp_mask].copy()
        fn_examples = df[fn_mask].copy()

        return fp_examples, fn_examples

    return pd.DataFrame(), pd.DataFrame()


def calculate_f1(df):
    if 'recurring_group_id' not in df.columns:
        return 0.0

    # Ground Truth
    if 'isRecurring' not in df.columns:
        return 0.0

    y_true = df['isRecurring'].fillna(False).astype(bool)
    y_pred = df['recurring_group_id'] != -1

    # Calculate metrics
    tp = ((y_pred == True) & (y_true == True)).sum()
    fp = ((y_pred == True) & (y_true == False)).sum()
    fn = ((y_pred == False) & (y_true == True)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1


if __name__ == "__main__":
    pass
