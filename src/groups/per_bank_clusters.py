import logging
import argparse
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.metrics import classification_report, precision_recall_fscore_support

# Suppress Warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*n_jobs.*")

from common.config import ExperimentConfig, FieldConfig, EmbModel
from common.feature_processor import FeatProcParams
from common.embedder import EmbeddingService
from common.log_utils import setup_logging
from common.data import filter_unique_bank_variants
from pointwise.runner import ExpRunner
from pointwise.classifier import HybridModel
from groups.model_clusterer import ModelBasedClusterer
from common.data import FeatureSet

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Per-Bank Model Clustering Benchmark")
    parser.add_argument("--data_path", type=str, default="data/combined_transactions_flat.csv",
                        help="Path to the large dataset with 'bank_name'")
    parser.add_argument("--output_path", type=str, default="results/per_bank_clustering_results.csv",
                        help="Path to save the consolidated results CSV")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU for clustering")
    return parser.parse_args()


def run_per_bank_experiment():
    args = parse_args()
    setup_logging(Path("logs/"), "per_bank_clustering")

    # 1. Load Data
    logger.info(f"Loading data from {args.data_path}...")
    try:
        full_df = pd.read_csv(args.data_path, low_memory=False)
    except FileNotFoundError:
        logger.error(f"File not found: {args.data_path}")
        return

    field_config = FieldConfig()

    # 2. Clean & Filter Variants (No other filtering as requested)
    # This keeps only the "best" variant for each bank entity
    df_clean = filter_unique_bank_variants(full_df)

    # Basic validity check (dates/amounts must exist)
    df_clean = df_clean.dropna(subset=[field_config.date, field_config.amount, field_config.text, field_config.label])

    # Ensure label is int
    # Handle boolean strings or actual booleans
    if df_clean[field_config.label].dtype == 'object':
        df_clean[field_config.label] = df_clean[field_config.label].astype(str).str.lower() == 'true'
    df_clean[field_config.label] = df_clean[field_config.label].astype(int)

    # 3. Configuration (Constant across banks)
    exp_config = ExperimentConfig(epochs=5, batch_size=128)

    feat_params = FeatProcParams(
        use_cyclical_dates=True,
        use_continuous_amount=True,
        use_categorical_amount=False
    )

    emb_params = EmbeddingService.Params(model_name=EmbModel.MPNET)

    mlp_params = HybridModel.MlpHyperParams(
        mlp_hidden_layers=[128, 64],
        dropout_rate=0.25,
        text_projection_dim=64
    )

    # 4. Initialize Runner (Shared Embedder Cache)
    # We create a dummy runner first just to initialize the shared components
    base_runner = ExpRunner.create(exp_config, df_clean, emb_params, feat_params, mlp_params, field_config)

    # 5. Per-Bank Loop
    results_list = []
    unique_banks = df_clean['bank_name'].unique()
    logger.info(f"Found {len(unique_banks)} unique banks.")

    # CSV Header
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Bank Name':<30} | {'Rows':<6} | {'F1 (Clust)':<10} | {'Prec':<6} | {'Rec':<6} | {'F1 (Point)':<10}")
    print("-" * 90)

    TRAIN_SIZE = 500  # Minimum rows to attempt training

    for bank in unique_banks:
        bank_df = df_clean[df_clean['bank_name'] == bank].copy()
        n_rows = len(bank_df)

        if n_rows < TRAIN_SIZE:
            logger.info(f"Skipping {bank}: Too small ({n_rows} rows)")
            continue

        logger.info(f"\n>>> Processing Bank: {bank} ({n_rows} rows)")

        try:
            # --- A. Setup Runner for this Bank ---
            # We create a fresh runner per bank to handle its specific data split
            runner = ExpRunner.create(exp_config, bank_df, emb_params, feat_params, mlp_params, field_config)

            # --- B. Split (3-Way) ---
            # We need enough data for 3 splits. If too small, sklearn might complain.
            try:
                df_train, df_val, df_test = runner.create_train_val_test_split(test_size=0.2, val_size=0.2)
            except ValueError as e:
                logger.warning(f"  Split failed for {bank}: {e}")
                continue

            # --- C. Build Data & Train Pointwise ---
            logger.info(f"  Training Pointwise Model on {len(df_train)} rows...")
            train_fs, val_fs, test_fs, processor, meta = runner.build_data_three_way(df_train, df_val, df_test)

            # Train
            val_metrics, model = runner.run_experiment_and_return_model(train_fs, val_fs, meta)

            # Evaluate Pointwise on Test (Baseline)
            pt_test_metrics = runner.evaluate_model_on_set(model, test_fs)

            # --- D. Run Model-Based Clustering ---
            logger.info("  Running Clustering on Test Set...")
            clusterer = ModelBasedClusterer(
                model, processor,
                min_samples=2,
                use_gpu=args.gpu,
                voting_threshold=0.5
            )

            # Cluster Test Data
            # Since we have the exact FeatureSet and DF for test, we can run it directly
            # (ModelBasedClusterer handles the whole set logic internally if passed correctly)
            # BUT, ModelBasedClusterer.cluster_features expects a GLOBAL feature set and DF.
            # Since 'df_test' is the whole test set for this bank, we can pass it directly.

            res = clusterer.cluster_features(test_fs, df_test)

            # --- E. Collect Metrics ---
            cl_metrics = res.metrics  # {'precision': ..., 'recall': ..., 'f1': ...}

            row = {
                "bank_name": bank,
                "total_rows": n_rows,
                "test_rows": len(df_test),
                "clustering_f1": cl_metrics['f1'],
                "clustering_precision": cl_metrics['precision'],
                "clustering_recall": cl_metrics['recall'],
                "pointwise_f1": pt_test_metrics['f1'],
                "pointwise_auc": pt_test_metrics['roc_auc']
            }

            results_list.append(row)

            # Log to Console
            print(
                f"{bank:<30} | {n_rows:<6} | {row['clustering_f1']:.4f}     | {row['clustering_precision']:.4f} | {row['clustering_recall']:.4f} | {row['pointwise_f1']:.4f}")

            # Incremental Save
            pd.DataFrame(results_list).to_csv(args.output_path, index=False)

        except Exception as e:
            logger.error(f"Failed processing bank {bank}: {e}", exc_info=True)

    logger.info("=" * 50)
    logger.info(f"Benchmark complete. Results saved to {args.output_path}")


if __name__ == "__main__":
    run_per_bank_experiment()