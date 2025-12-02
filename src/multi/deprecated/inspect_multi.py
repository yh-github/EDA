import argparse
import logging
import pandas as pd
import numpy as np
from multi.config import MultiExpConfig, MultiFieldConfig
from multi.deprecated.inference import MultiPredictor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("inspector")


def inspect_clusters(data_path, model_path):
    """
    Runs inference on a subset of data and prints the discovered patterns.
    """
    # 1. Setup
    config = MultiExpConfig()
    fc = MultiFieldConfig()

    # 2. Load Data
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Take a random sample of accounts to keep it fast
    unique_accs = df[fc.accountId].unique()
    if len(unique_accs) > 100:
        sample_accs = np.random.choice(unique_accs, 100, replace=False)
        df = df[df[fc.accountId].isin(sample_accs)].copy()
        logger.info(f"Sampled down to {len(df)} transactions (100 accounts) for inspection.")

    # 3. Run Inference
    logger.info("Loading model and running inference...")
    predictor = MultiPredictor(model_path, config)
    results = predictor.predict(df)

    if results.empty:
        logger.warning("No predictions returned!")
        return

    # 4. Analysis
    logger.info("\n" + "=" * 60)
    logger.info("  🔍  CLUSTER INSPECTION REPORT")
    logger.info("=" * 60)

    # Filter for only predicted recurring transactions
    recurring = results[results['pred_isRecurring'] == True].copy()

    if recurring.empty:
        logger.warning("Model predicted NO recurring transactions in this sample.")
        return

    # Group by the Predicted Pattern ID
    clusters = recurring.groupby('pred_patternId')

    # Sort clusters by confidence (mean prob)
    # FIX: Use fields from config instead of hardcoded strings
    cluster_stats = clusters.agg({
        'pred_recurring_prob': 'mean',
        fc.amount: 'mean',
        fc.text: lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
        fc.date: 'count'
    }).rename(columns={fc.date: 'size', fc.text: 'description'})

    top_clusters = cluster_stats.sort_values('pred_recurring_prob', ascending=False).head(15)

    logger.info(f"\nFound {len(clusters)} unique patterns in sample.")
    logger.info("\n🏆 TOP 15 HIGH-CONFIDENCE CLUSTERS")
    logger.info(f"{'Prob':<8} | {'Size':<5} | {'Avg Amt':<10} | {'Description'}")
    logger.info("-" * 60)

    for pid, row in top_clusters.iterrows():
        # Ensure description is string
        desc = str(row['description'])
        # Truncate if too long
        if len(desc) > 50: desc = desc[:47] + "..."

        logger.info(
            f"{row['pred_recurring_prob']:.4f}   | {row['size']:<5} | {row[fc.amount]:<10.2f} | {desc}")

    # 5. Inspect a specific complex case (optional)
    # Find a cluster with size > 5 to see if it grouped variations
    large_clusters = cluster_stats[cluster_stats['size'] > 4]
    if not large_clusters.empty:
        best_large_pid = large_clusters['pred_recurring_prob'].idxmax()

        logger.info("\n" + "-" * 60)
        logger.info(f"🕵️  DEEP DIVE: Cluster {best_large_pid}")
        logger.info("-" * 60)

        txns = recurring[recurring['pred_patternId'] == best_large_pid].sort_values(fc.date)
        for _, row in txns.iterrows():
            logger.info(f"{row[fc.date]} | {row[fc.amount]:>8.2f} | {row[fc.text]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--model")
    args = parser.parse_args()

    inspect_clusters(args.data, args.model)