import argparse
import logging
import pandas as pd
import numpy as np
import torch

from common.config import get_device
from multi.config import MultiFieldConfig
from multi.data import get_dataloader
from multi.reload_utils import load_model_for_eval, load_data_for_config

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("report_gen")


def generate_report(model, df, loader, config, output_path):
    logger.info(f"Generating report for {len(df)} transactions...")

    device = get_device()

    model.eval()
    model.to(device)

    fc = MultiFieldConfig()

    results = {
        'original_index': [],
        'pred_prob': [],
        'pred_is_recurring': [],
        'pred_cycle': []
    }

    with torch.no_grad():
        for batch in loader:
            batch_gpu = {k: v.to(device) for k, v in batch.items()}

            # Forward
            _, cycle_logits, _ = model(batch_gpu)

            # Predictions
            probs = torch.softmax(cycle_logits, dim=-1)

            p_noise = probs[:, :, 0]
            p_recurring = 1.0 - p_noise

            preds_cls = torch.argmax(cycle_logits, dim=-1)

            # Collect data
            mask = batch['padding_mask'].bool().cpu().numpy()
            indices = batch['original_index'].cpu().numpy()
            p_rec_np = p_recurring.cpu().numpy()
            preds_cls_np = preds_cls.cpu().numpy()

            valid_indices = indices[mask]
            valid_probs = p_rec_np[mask]
            valid_preds = preds_cls_np[mask]

            results['original_index'].extend(valid_indices)
            results['pred_prob'].extend(valid_probs)
            results['pred_is_recurring'].extend(valid_preds > 0)
            results['pred_cycle'].extend(valid_preds)

    # Convert results to DataFrame
    pred_df = pd.DataFrame(results)
    pred_df.set_index('original_index', inplace=True)

    # Join with original Data
    full_report = df.copy()
    full_report = full_report.join(pred_df, how='left')

    # Fill NaNs for filtered rows (assume not recurring)
    full_report['pred_prob'] = full_report['pred_prob'].fillna(0.0)
    full_report['pred_is_recurring'] = full_report['pred_is_recurring'].fillna(False).astype(bool)

    # Determine Mistake Type
    y_true = full_report[fc.label].astype(bool)
    y_pred = full_report['pred_is_recurring']

    conditions = [
        (y_true == True) & (y_pred == True),
        (y_true == False) & (y_pred == False),
        (y_true == False) & (y_pred == True),
        (y_true == True) & (y_pred == False)
    ]
    choices = ['TP', 'TN', 'FP', 'FN']
    full_report['result_type'] = np.select(conditions, choices, default='Error')

    # Select columns for the final report
    cols = [
        fc.accountId, fc.date, fc.amount, fc.text,
        fc.label, 'patternId',  # Original Labels
        'pred_is_recurring', 'pred_prob', 'result_type'  # Model Predictions
    ]

    final_df = full_report[cols].sort_values([fc.accountId, fc.date])

    logger.info(f"Saving report to {output_path}...")
    final_df.to_csv(output_path, index=False)

    return final_df


def analyze_calibration(df):
    """
    Analyzes how accuracy changes with confidence.
    """
    logger.info("\n--- Calibration Analysis ---")

    # Bin probabilities into deciles
    df['prob_bin'] = pd.cut(df['pred_prob'], bins=np.linspace(0, 1, 11), labels=False)

    stats = df.groupby('prob_bin').agg({
        'result_type': 'count',
        'pred_prob': 'mean'
    }).rename(columns={'result_type': 'count', 'pred_prob': 'avg_conf'})

    high_conf_fp = df[(df['pred_prob'] > 0.9) & (df[MultiFieldConfig.label] == 0)]
    high_conf_fn = df[(df['pred_prob'] < 0.1) & (df[MultiFieldConfig.label] == 1)]

    total_fp = len(df[df['result_type'] == 'FP'])
    total_fn = len(df[df['result_type'] == 'FN'])

    if total_fp > 0:
        logger.info(
            f"Total FP: {total_fp}. High Confidence (>90%) FP: {len(high_conf_fp)} ({len(high_conf_fp) / total_fp:.1%} of FPs)")

    if total_fn > 0:
        logger.info(
            f"Total FN: {total_fn}. High Confidence (<10%) FN: {len(high_conf_fn)} ({len(high_conf_fn) / total_fn:.1%} of FNs)")

    if len(high_conf_fp) > 0:
        logger.info("\nHigh confidence FPs are excellent candidates for Label Errors.")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="cache/data")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="mistake_report.csv")
    args = parser.parse_args()

    # 1. Load Model
    try:
        model, config, device = load_model_for_eval(args.model_path, args.device)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # 2. Load Data
    try:
        _, val_df, test_df = load_data_for_config(config)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # 3. Create Loader
    logger.info("Initializing Test Loader...")
    loader = get_dataloader(test_df, config, shuffle=False, n_workers=0)

    # 4. Generate Report
    report_df = generate_report(model, test_df, loader, config, args.output)

    # 5. Run Calibration Check
    analyze_calibration(report_df)


if __name__ == "__main__":
    main()