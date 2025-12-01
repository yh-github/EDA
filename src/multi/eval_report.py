import argparse
import logging
import pickle
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from multi.config import MultiExpConfig, MultiFieldConfig
from multi.encoder import TransactionTransformer
from multi.tune_multi import get_data_cache_path
from multi.data import get_dataloader

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("report_gen")


def load_cached_data(config: MultiExpConfig):
    data_cache_path = get_data_cache_path(
        random_state=config.random_state, downsample=config.downsample
    )
    if not data_cache_path.exists():
        raise FileNotFoundError(f"Cache directory {data_cache_path} does not exist.")

    with open(data_cache_path, 'rb') as f:
        data = pickle.load(f)
    return data['train'], data['val'], data['test']


def generate_report(model, df, loader, config, output_path):
    logger.info(f"Generating report for {len(df)} transactions...")

    model.eval()
    device = config.device
    fc = MultiFieldConfig()

    # We need to map predictions back to the original dataframe.
    # The loader returns 'original_index' which maps to the dataframe index (if we use the aligned df).
    # However, to be safe and robust, we will collect all predictions into a dictionary keyed by (accountId, original_index)
    # or just use the index if we trust it.

    # Let's trust the 'original_index' from the loader as we fixed it in data.py.

    # Store results
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
            # Probability of being ANY recurring class (sum of classes 1..N)
            # recur_prob = probs[:, :, 1:].sum(dim=-1)
            # Actually, let's stick to the max prob class logic or sum of recurring?
            # The model is trained with CrossEntropy.
            # Class 0 = Noise.
            # Prob(Recurring) = 1.0 - Prob(Class 0)

            p_noise = probs[:, :, 0]
            p_recurring = 1.0 - p_noise

            preds_cls = torch.argmax(cycle_logits, dim=-1)

            # Collect data
            # Flatten batch and sequence
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
    # We use a left join to keep the original data structure
    # Note: The loader only iterates over transactions with direction != 0.
    # Rows filtered out by the dataset will have NaN predictions.

    full_report = df.copy()
    full_report = full_report.join(pred_df, how='left')

    # Fill NaNs for filtered rows (assume not recurring)
    full_report['pred_prob'] = full_report['pred_prob'].fillna(0.0)
    full_report['pred_is_recurring'] = full_report['pred_is_recurring'].fillna(False).astype(bool)

    # Determine Mistake Type
    # Ground Truth
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

    # Calculate Accuracy per bin
    # We need to be careful:
    # For low prob bins (e.g. 0.1), "correct" means the label is 0 (Noise).
    # For high prob bins (e.g. 0.9), "correct" means the label is 1 (Recurring).

    # Let's verify 'High Confidence' mistakes specifically.
    # A "High Confidence Mistake" is:
    # 1. Prob > 0.9 but Label = 0 (High Conf FP)
    # 2. Prob < 0.1 but Label = 1 (High Conf FN)

    high_conf_fp = df[(df['pred_prob'] > 0.9) & (df[MultiFieldConfig.label] == 0)]
    high_conf_fn = df[(df['pred_prob'] < 0.1) & (df[MultiFieldConfig.label] == 1)]

    total_fp = len(df[df['result_type'] == 'FP'])
    total_fn = len(df[df['result_type'] == 'FN'])

    logger.info(
        f"Total FP: {total_fp}. High Confidence (>90%) FP: {len(high_conf_fp)} ({len(high_conf_fp) / total_fp:.1%} of FPs)")
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

    # Load Model
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    config.device = device

    model = TransactionTransformer(config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # Load Data
    _, val_df, test_df = load_cached_data(config)

    # Create Loader for TEST set (usually what we want to verify)
    loader = get_dataloader(test_df, config, shuffle=False, n_workers=0)

    # Generate Report
    report_df = generate_report(model, test_df, loader, config, args.output)

    # Run Calibration Check
    analyze_calibration(report_df)


if __name__ == "__main__":
    main()