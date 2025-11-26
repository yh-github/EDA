import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from common.config import ExperimentConfig, FieldConfig, EmbModel
from common.feature_processor import FeatProcParams
from glob.classifier import HybridModel
from common.embedder import EmbeddingService
from glob.runner import ExpRunner
from common.log_utils import setup_logging

# Setup Logging
logger = logging.getLogger("bank_benchmark")


def parse_args():
    parser = argparse.ArgumentParser(description="Bank Benchmark Experiment")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/combined_transactions_flat.csv",
        help="Path to the new data file containing 'bank_name' column."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/bank_benchmark_results.csv",
        help="Path to save the results CSV."
    )
    return parser.parse_args()


def get_constant_configs():
    """
    Defines the constant parameters for the experiment.
    These are chosen based on the 'winner' params from previous tuning runs (e.g. tune1/tune2).
    """
    # 1. Experiment Config
    exp_config = ExperimentConfig(
        epochs=10,
        batch_size=128,
        learning_rate=1e-3,
        early_stopping_patience=3
    )

    # 2. Embedding Params (MPNET is generally robust)
    emb_params = EmbeddingService.Params(
        model_name=EmbModel.MPNET,
        batch_size=128
    )

    # 3. Feature Processing Params
    # Using a robust set of features based on previous exploration
    feat_params = FeatProcParams(
        use_cyclical_dates=True,
        use_categorical_dates=True,
        use_continuous_amount=True,
        use_categorical_amount=False,
        use_is_positive=False,  # We are filtering for amount > 0 anyway
        k_top=20,
        n_bins=20
    )

    # 4. MLP Model Params
    model_params = HybridModel.MlpHyperParams(
        mlp_hidden_layers=[128, 64],
        dropout_rate=0.25,
        text_projection_dim=None
    )

    return exp_config, emb_params, feat_params, model_params


def clean_and_filter_data(df: pd.DataFrame, field_config: FieldConfig) -> pd.DataFrame:
    """
    1. Remove digits from bank_name.
    2. Filter amount > 0.
    3. Filter accounts with row count >= 10.
    """
    logger.info(f"Initial data size: {len(df)}")

    # Ensure bank_name exists
    bank_name = field_config.bank_name
    if bank_name not in df.columns:
        raise ValueError("Column 'bank_name' missing from data file.")

    # 1. Remove digits from bank_name
    df[bank_name] = df[bank_name].astype(str).str.replace(r'\d+', '', regex=True).str.strip()

    # 2. Filter Amount > 0
    df = df[df[field_config.amount] > 0].copy()
    logger.info(f"After amount > 0 filter: {len(df)}")

    # 3. Filter Accounts with count >= 10
    acc_counts = df[field_config.accountId].value_counts()
    valid_accounts = acc_counts[acc_counts >= 10].index
    df = df[df[field_config.accountId].isin(valid_accounts)].copy()
    logger.info(f"After account count >= 10 filter: {len(df)} rows, {len(valid_accounts)} accounts")

    return df


def run_bank_benchmark():
    args = parse_args()
    setup_logging(Path("logs/"), "bank_benchmark")

    # --- 1. Load Configurations ---
    field_config = FieldConfig()
    exp_config, emb_params, feat_params, model_params = get_constant_configs()

    # --- 2. Load and Preprocess Data ---
    logger.info(f"Loading data from {args.data_path}...")
    try:
        full_df = pd.read_csv(args.data_path)
    except FileNotFoundError:
        logger.error(f"File not found: {args.data_path}")
        return

    df_clean = clean_and_filter_data(full_df, field_config)

    # --- 3. Initialize Runner ---
    # We create one runner instance to manage embeddings and shared logic.
    # We pass df_clean just to satisfy init, but we will handle splits manually.
    runner = ExpRunner.create(
        exp_params=exp_config,
        full_df=df_clean,
        emb_params=emb_params,
        feat_proc_params=feat_params,
        model_params=model_params,
        field_config=field_config
    )

    # --- 4. Per-Bank Loop ---
    results_list = []
    unique_banks = df_clean['bank_name'].unique()
    logger.info(f"Found {len(unique_banks)} unique banks: {unique_banks}")

    # Prepare CSV header by creating empty file (or overwrite)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Bank Name':<30} | {'F1':<6} | {'Prec':<6} | {'Rec':<6} | {'Best F1':<8} | {'Thresh':<6}")
    print("-" * 80)

    for bank in unique_banks:
        bank_df = df_clean[df_clean['bank_name'] == bank]
        unique_accounts = bank_df[field_config.accountId].unique()
        n_accounts = len(unique_accounts)

        # Check sufficiency
        if n_accounts < 2000:
            logger.warning(f"Skipping bank '{bank}': Only {n_accounts} accounts (need 2000 for 1k/1k split).")
            continue

        # Sample Accounts
        # Using fixed seed for reproducibility of the split
        rng = np.random.RandomState(exp_config.random_state)
        shuffled_accs = rng.permutation(unique_accounts)

        train_accs = shuffled_accs[:1000]
        test_accs = shuffled_accs[1000:2000]

        df_train = bank_df[bank_df[field_config.accountId].isin(train_accs)]
        df_test = bank_df[bank_df[field_config.accountId].isin(test_accs)]

        logger.info(f"Training on bank: {bank} (Train: {len(df_train)} rows, Test: {len(df_test)} rows)")

        try:
            # Prepare Data (Embed, Process, etc.) using ExpRunner's helper
            train_fs, test_fs, _, meta = runner.build_data(df_train, df_test)

            # Train and Evaluate
            metrics = runner.run_experiment(train_fs, test_fs, meta)

            # Collect Results
            row = {
                "bank_name": bank,
                "train_rows": len(df_train),
                "test_rows": len(df_test),
                "f1": metrics.get("f1", 0),
                "precision": metrics.get("final_precision", 0),
                # Metrics might need adjustment depending on what run_experiment returns
                "recall": metrics.get("final_recall", 0),
                "roc_auc": metrics.get("roc_auc", 0),
                "best_f1": metrics.get("val_best_f1", 0),
                "best_threshold": metrics.get("val_best_threshold", 0)
            }

            # Print to screen
            print(
                f"{bank:<30} | {row['f1']:.4f} | {metrics.get('accuracy', 0):.4f} | n/a    | {row['best_f1']:.4f}   | {row['best_threshold']:.4f}")

            results_list.append(row)

            # Incremental Save
            pd.DataFrame(results_list).to_csv(args.output_path, index=False)

        except Exception as e:
            logger.error(f"Failed processing bank {bank}: {e}", exc_info=True)

    logger.info("=" * 50)
    logger.info(f"Benchmark complete. Results saved to {args.output_path}")


if __name__ == "__main__":
    run_bank_benchmark()