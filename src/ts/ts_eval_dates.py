import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import logging
from common.config import FieldConfig
from ts_robust_analyzer import RobustTSAnalyzer, TSConfig
from common.log_utils import setup_logging

setup_logging(Path('logs/'), "eval_ts_dates")
logger = logging.getLogger(__name__)


def evaluate_account(df: pd.DataFrame, fc: FieldConfig, analyzer: RobustTSAnalyzer):
    """
    Compares predicted dates vs actual 'isRecurring' dates for one account.
    Returns: (TP, FP, FN)
    """
    # 1. Get Ground Truth Dates
    # Where isRecurring == True
    true_dates = pd.to_datetime(df[df[fc.label] == True][fc.date]).dt.normalize().unique()
    true_dates = np.sort(true_dates)

    if len(true_dates) == 0:
        return 0, 0, 0  # No recurring events to find

    # 2. Get Predictions
    # We pass the WHOLE dataframe. The analyzer's job is to find the signal amidst noise.
    pred_dates = analyzer.predict_dates(df)
    pred_dates = np.sort(pred_dates)

    # 3. Match (Allowing Jitter)
    # Simple greedy matching
    tp = 0
    fp = 0
    # fn = 0

    tolerance = timedelta(days=analyzer.cfg.jitter_tolerance)

    # Use a set for fast lookups of matched ground truth
    matched_truth = set()

    for pred in pred_dates:
        # Is there a match in truth?
        # Check window [pred - tol, pred + tol]
        match_found = False
        for truth in true_dates:
            if truth in matched_truth: continue

            if abs(pred - truth) <= tolerance:
                match_found = True
                matched_truth.add(truth)
                tp += 1
                break  # Found a match for this prediction

        if not match_found:
            fp += 1

    fn = len(true_dates) - len(matched_truth)

    return tp, fp, fn


def main():
    fc = FieldConfig()
    logger.info("Loading data...")
    df = pd.read_csv("data/rec_data2.csv")
    df = df.dropna(subset=[fc.date, fc.amount, fc.text])

    # Config to Test
    cfg = TSConfig(
        signal_mode='binary',
        max_harmonics=5,
        spectral_threshold=0.15,
        detection_threshold=0.3
    )
    analyzer = RobustTSAnalyzer(fc, cfg)

    total_tp, total_fp, total_fn = 0, 0, 0

    accounts = df[fc.accountId].unique()
    logger.info(f"Evaluating {len(accounts)} accounts...")

    for acc_id in accounts:
        acc_df = df[df[fc.accountId] == acc_id]
        if len(acc_df) < 14: continue

        tp, fp, fn = evaluate_account(acc_df, fc, analyzer)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Final Stats
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    logger.info("=" * 40)
    logger.info(f"RESULTS (Spectral Analysis)")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info("=" * 40)


if __name__ == "__main__":
    main()
