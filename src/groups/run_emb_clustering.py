import logging
import pandas as pd
import argparse
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

from common.config import FieldConfig, EmbModel, ExperimentConfig
from common.data import create_train_val_test_split
from common.feature_processor import FeatProcParams
from common.embedder import EmbeddingService
from common.log_utils import setup_logging
from groups.emb_clusterer import EmbClusterer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Embedding Clustering on Standard Splits")
    parser.add_argument("--data_path", type=str, default="data/rec_data2.csv", help="Path to dataset")
    parser.add_argument("--output_path", type=str, default="results/emb_clustering_predictions.csv")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration for UMAP/HDBSCAN")
    return parser.parse_args()


def evaluate_subset(
        set_name: str,
        df: pd.DataFrame,
        clusterer: EmbClusterer,
        field_config: FieldConfig
) -> pd.DataFrame:
    """
    Runs clustering on a specific data subset (Val/Test) and computes metrics.
    """
    logger.info(
        f"\n--- Processing {set_name} Set ({len(df)} rows, {df[field_config.accountId].nunique()} accounts) ---")

    results = []

    # Process each account in this split
    for acc_id, acc_df in df.groupby(field_config.accountId):
        try:
            res = clusterer.cluster_account(acc_df)
            results.append(res.clustered_df)
        except Exception as e:
            logger.error(f"Error processing account {acc_id}: {e}")
            # Fallback: predict 0
            fallback = acc_df.copy()
            fallback['prediction'] = 0
            fallback['cluster_label'] = -1
            results.append(fallback)

    if not results:
        logger.warning(f"No results for {set_name} set.")
        return pd.DataFrame()

    final_df = pd.concat(results, ignore_index=True)

    # Metrics
    y_true = final_df[field_config.label].astype(int).values
    y_pred = final_df['prediction'].values

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

    logger.info(f"\n>>> METRICS: {set_name}")
    logger.info(f"Precision : {p:.4f}")
    logger.info(f"Recall    : {r:.4f}")
    logger.info(f"F1 Score  : {f1:.4f}")

    return final_df


def run_global_clustering():
    args = parse_args()
    setup_logging(Path("logs/"), "emb_clustering_split")

    # 1. Load Data
    logger.info(f"Loading data from {args.data_path}...")
    try:
        df = pd.read_csv(args.data_path)
    except FileNotFoundError:
        logger.error(f"File not found: {args.data_path}")
        return

    field_config = FieldConfig()

    # Clean Data
    df_clean = df.dropna(subset=[field_config.date, field_config.amount, field_config.text, field_config.label]).copy()
    df_clean[field_config.label] = df_clean[field_config.label].astype(int)

    # 2. Create Splits (Matches tft_tune5 logic)
    exp_config = ExperimentConfig()
    logger.info(f"Creating splits with random_state={exp_config.random_state}...")

    _, val_df, test_df = create_train_val_test_split(
        test_size=0.2,
        val_size=0.2,
        full_df=df_clean,
        random_state=exp_config.random_state,
        field_config=field_config
    )

    # 3. Setup Clusterer
    emb_service = EmbeddingService.create(
        EmbeddingService.Params(model_name=EmbModel.MPNET, batch_size=512)
    )

    feat_params = FeatProcParams(
        use_continuous_amount=True,
        use_cyclical_dates=True,
        use_categorical_dates=False,
        use_categorical_amount=False,
        use_behavioral_features=False
    )

    clusterer = EmbClusterer(
        field_config=field_config,
        feat_params=feat_params,
        emb_service=emb_service,
        min_samples=3,
        umap_components=5,
        umap_neighbors=15,
        cluster_epsilon=0.1,
        use_gpu=args.gpu
    )

    # 4. Evaluate on Validation and Test
    val_results_df = evaluate_subset("VALIDATION", val_df, clusterer, field_config)
    test_results_df = evaluate_subset("TEST", test_df, clusterer, field_config)

    # 5. Save Combined Results
    if not val_results_df.empty and not test_results_df.empty:
        val_results_df['split'] = 'validation'
        test_results_df['split'] = 'test'

        all_results = pd.concat([val_results_df, test_results_df], ignore_index=True)

        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        all_results.to_csv(output_path, index=False)
        logger.info(f"\nSaved all predictions to {output_path}")


if __name__ == "__main__":
    run_global_clustering()