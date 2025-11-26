import logging
import pandas as pd

import argparse
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, classification_report

from common.config import FieldConfig, EmbModel
from common.feature_processor import FeatProcParams
from common.embedder import EmbeddingService
from common.log_utils import setup_logging
from groups.emb_clusterer import EmbClusterer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Embedding Clustering on All Accounts")
    parser.add_argument("--data_path", type=str, default="data/rec_data2.csv", help="Path to dataset")
    parser.add_argument("--output_path", type=str, default="results/emb_clustering_predictions.csv")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration for UMAP/HDBSCAN")
    return parser.parse_args()


def run_global_clustering():
    args = parse_args()
    setup_logging(Path("logs/"), "emb_clustering_global")

    # 1. Load Data
    logger.info(f"Loading data from {args.data_path}...")
    try:
        df = pd.read_csv(args.data_path)
    except FileNotFoundError:
        logger.error(f"File not found: {args.data_path}")
        return

    field_config = FieldConfig()

    # Filter for valid rows
    df_clean = df.dropna(subset=[field_config.date, field_config.amount, field_config.text, field_config.label]).copy()

    # Ensure label is int (0/1) for metrics
    df_clean[field_config.label] = df_clean[field_config.label].astype(int)

    logger.info(f"Total Transactions: {len(df_clean)}")
    logger.info(f"Unique Accounts: {df_clean[field_config.accountId].nunique()}")

    # 2. Setup Configuration
    # We use a robust feature set for clustering
    feat_params = FeatProcParams(
        use_continuous_amount=True,
        use_cyclical_dates=True,
        use_categorical_dates=False,  # UMAP prefers continuous
        use_categorical_amount=False,
        use_behavioral_features=False  # We are clustering raw txns, not user profiles
    )

    # Initialize Embedder (Shared across all accounts to use cache)
    logger.info("Initializing Embedding Service...")
    emb_service = EmbeddingService.create(
        EmbeddingService.Params(model_name=EmbModel.MPNET, batch_size=512)
    )

    # 3. Initialize Clusterer
    # We pass the shared embedder. The clusterer will use it for each account.
    clusterer = EmbClusterer(
        field_config=field_config,
        feat_params=feat_params,
        emb_service=emb_service,
        min_samples=3,  # Minimum cluster size
        umap_components=5,  # Dimensions for density search
        umap_neighbors=15,  # Local neighborhood size
        cluster_epsilon=0.1,  # DBSCAN density threshold
        use_gpu=args.gpu
    )

    # 4. Processing Loop
    all_results_df = []

    logger.info("Starting Clustering Loop...")

    # Group by account for independent processing
    # tqdm provides a progress bar
    for acc_id, acc_df in df_clean.groupby(field_config.accountId):
        try:
            # Run clustering for this specific account
            result = clusterer.cluster_account(acc_df)

            # Collect the dataframe which now has 'prediction' and 'cluster_label' columns
            all_results_df.append(result.clustered_df)

        except Exception as e:
            logger.error(f"Error processing account {acc_id}: {e}", exc_info=True)
            # In case of error, we should probably append the original df with 0 predictions
            # to maintain true recall calculation
            error_df = acc_df.copy()
            error_df['prediction'] = 0
            error_df['cluster_label'] = -1
            all_results_df.append(error_df)

    # 5. Aggregation
    if not all_results_df:
        logger.error("No results generated.")
        return

    final_df = pd.concat(all_results_df, ignore_index=True)

    # 6. Global Metrics Calculation
    y_true = final_df[field_config.label].values
    y_pred = final_df['prediction'].values

    logger.info("\n" + "=" * 50)
    logger.info("GLOBAL CLUSTERING METRICS (All Accounts)")
    logger.info("=" * 50)

    # Binary metrics
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    logger.info(f"Total Rows Evaluated : {len(final_df)}")
    logger.info(f"Precision            : {p:.4f}")
    logger.info(f"Recall               : {r:.4f}")
    logger.info(f"F1 Score             : {f1:.4f}")

    # Detailed Report
    report = classification_report(y_true, y_pred, target_names=['Non-Recurring', 'Recurring'])
    print("\n" + report)

    # 7. Save Results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logger.info(f"Detailed predictions saved to {output_path}")


if __name__ == "__main__":
    run_global_clustering()