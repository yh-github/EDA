import logging
import argparse
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, classification_report

# Suppress Warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*n_jobs.*")

from common.config import ExperimentConfig, FieldConfig, EmbModel
from common.feature_processor import FeatProcParams
from common.embedder import EmbeddingService
from common.log_utils import setup_logging
from common.data import filter_unique_bank_variants, FeatureSet, deduplicate
from pointwise.runner import ExpRunner
from pointwise.classifier import HybridModel
# Ensure your model_clusterer.py has the 'enable_recovery' update!
from groups.model_clusterer2 import ModelBasedClusterer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Mixed Bank Training & Clustering")
    parser.add_argument("--data_path", type=str, default="data/combined_transactions_flat.csv")
    parser.add_argument("--output_path", type=str, default="results/mixed_model_clustering.csv")
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def evaluate_per_bank(final_df: pd.DataFrame, field_config: FieldConfig) -> pd.DataFrame:
    """
    Calculates Precision/Recall/F1 for each bank in the test set.
    """
    bank_metrics = []

    # Iterate over banks present in the Test Set
    for bank, group in final_df.groupby('bank_name'):
        y_true = group[field_config.label].astype(int).values
        y_pred = group['prediction'].values

        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

        bank_metrics.append({
            "bank_name": bank,
            "test_rows": len(group),
            "precision": p,
            "recall": r,
            "f1": f1
        })

    return pd.DataFrame(bank_metrics).sort_values('f1', ascending=False)


def run_mixed_experiment():
    args = parse_args()
    setup_logging(Path("logs/"), "mixed_training")

    # 1. Load Data
    logger.info(f"Loading data from {args.data_path}...")
    try:
        full_df = pd.read_csv(args.data_path, low_memory=False)
    except FileNotFoundError:
        logger.error(f"File not found: {args.data_path}")
        return

    field_config = FieldConfig()

    # 2. Filter Variants
    logger.info("Filtering unique bank variants...")
    df_clean = deduplicate(full_df)

    # Essential sanity checks
    df_clean = df_clean.dropna(subset=[
        field_config.date, field_config.amount, field_config.text, field_config.label
    ])

    # Normalize Labels
    if df_clean[field_config.label].dtype == 'object':
        df_clean[field_config.label] = df_clean[field_config.label].astype(str).str.lower() == 'true'
    df_clean[field_config.label] = df_clean[field_config.label].astype(int)

    # --- CRITICAL FIX: ACCOUNT-BASED SAMPLING ---
    # We must keep the *entire history* for a subset of users.
    logger.info("Downsampling dataset to 33% of ACCOUNTS...")
    unique_accounts = df_clean[field_config.accountId].unique()

    # Randomly select 33% of account IDs
    rng = np.random.default_rng(112025)
    selected_accounts = rng.choice(unique_accounts, size=int(len(unique_accounts) * 0.33), replace=False)

    # Filter dataframe to keep ONLY rows from selected accounts
    df_clean = df_clean[df_clean[field_config.accountId].isin(selected_accounts)].copy()

    logger.info(
        f"Dataset Ready: {len(df_clean)} rows ({len(selected_accounts)} accounts) across {df_clean['bank_name'].nunique()} banks.")

    # 3. Configuration
    exp_config = ExperimentConfig(epochs=5, batch_size=256)

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

    runner = ExpRunner.create(exp_config, df_clean, emb_params, feat_params, mlp_params, field_config)

    # 4. Global Split (Train/Val/Test)
    logger.info("Creating Global Train/Val/Test Split...")
    df_train, df_val, df_test = runner.create_train_val_test_split(test_size=0.2, val_size=0.2)

    logger.info(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # 5. Build Data & Train
    logger.info(">>> STEP 1: Training Global Pointwise Model...")
    train_fs, val_fs, test_fs, processor, meta = runner.build_data_three_way(df_train, df_val, df_test)

    # Train
    val_metrics, model = runner.run_experiment_and_return_model(train_fs, val_fs, meta)
    logger.info(f"Global Validation Metrics: {val_metrics}")

    # Evaluate Pointwise Baseline on Test
    pt_test_metrics = runner.evaluate_model_on_set(model, test_fs)
    logger.info(f"Global Pointwise Test Metrics: {pt_test_metrics}")

    # 6. Run Model-Based Clustering (Test Set)
    logger.info(">>> STEP 2: Running Model-Based Clustering on Global Test Set...")

    # Initialize Clusterer WITH Recovery
    clusterer = ModelBasedClusterer(
        model, processor,
        min_samples=2,
        use_gpu=args.gpu,
        voting_threshold=0.4,
        anchoring_threshold=0.85,
        enable_recovery=True,
        recovery_distance_threshold=2.5
    )

    results = []
    test_indices = df_test.groupby(field_config.accountId).indices
    total_accounts = len(test_indices)

    # Manual loop for logging
    for i, (acc_id, idxs) in enumerate(test_indices.items()):
        if i % 1000 == 0 and i > 0:
            logger.info(f"  Clustering Progress: {i}/{total_accounts} accounts processed...")

        indices = np.sort(idxs)

        sub_fs = FeatureSet(
            X_text=test_fs.X_text[indices],
            X_continuous=test_fs.X_continuous[indices],
            X_categorical=test_fs.X_categorical[indices],
            y=test_fs.y[indices]
        )

        sub_df = df_test.iloc[indices].copy()

        res = clusterer.cluster_features(sub_fs, sub_df)
        results.append(res.clustered_df)

    # 7. Evaluation & Breakdown
    final_df = pd.concat(results)

    logger.info("\n" + "=" * 50)
    logger.info("GLOBAL CLUSTERING RESULTS")
    logger.info("=" * 50)

    y_true = final_df[field_config.label].astype(int)
    y_pred = final_df['prediction']

    # Log detailed report
    report = classification_report(y_true, y_pred)
    logger.info(f"\n{report}")

    # Per-Bank Breakdown
    logger.info("Calculating Per-Bank Metrics...")
    bank_stats = evaluate_per_bank(final_df, field_config)

    log_buffer = ["\n" + "=" * 80]
    log_buffer.append(f"{'Bank Name':<30} | {'Rows':<8} | {'F1':<8} | {'Prec':<8} | {'Rec':<8}")
    log_buffer.append("-" * 80)

    for _, row in bank_stats.iterrows():
        log_buffer.append(
            f"{row['bank_name']:<30} | {row['test_rows']:<8} | {row['f1']:.4f}   | {row['precision']:.4f}   | {row['recall']:.4f}")

    log_buffer.append("=" * 80)

    # Write table to log
    logger.info("\n".join(log_buffer))

    # Save full results
    final_df.to_csv(args.output_path, index=False)
    logger.info(f"Predictions saved to {args.output_path}")


if __name__ == "__main__":
    run_mixed_experiment()