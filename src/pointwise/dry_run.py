import argparse
import logging
from pathlib import Path
import pandas as pd
from classifier import HybridModel
from common.config import ExperimentConfig, FieldConfig, EmbModel
from common.embedder import EmbeddingService
from common.feature_processor import FeatProcParams
from common.log_utils import setup_logging
from runner import ExpRunner


def main():
    parser = argparse.ArgumentParser(description="Dry run for the experiment setup.")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the rec_data2.csv file."
    )
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("--- STARTING SYSTEM DRY RUN ---")
    logger.info("=" * 50)

    # --- 1. Config Initialization ---
    logger.info("--- 1. Initializing Configs ---")
    exp_config = ExperimentConfig()  # Uses defaults (10 epochs, etc.)
    field_config = FieldConfig()
    feat_params = FeatProcParams(n_bins=20, k_top=50)  # Use feature-rich params
    emb_params = EmbeddingService.Params(model_name=EmbModel.ALBERT)
    mlp_params = HybridModel.MlpHyperParams()  # Use default MLP params

    logger.info(f"ExperimentConfig: {exp_config}")
    logger.info(f"FeatProcParams: {feat_params}")
    logger.info(f"EmbModel: {emb_params.model_name}")
    logger.info(f"MlpParams: {mlp_params}")
    logger.info("-" * 50)

    # --- 2. Data Loading ---
    logger.info(f"--- 2. Loading Data from {args.data_path} ---")
    try:
        df = pd.read_csv(args.data_path)
        df_cleaned = df.dropna(
            subset=[
                field_config.date,
                field_config.amount,
                field_config.text,
                field_config.label
            ]
        )
        logger.info(f"Loaded {len(df)} rows, {len(df_cleaned)} after cleaning.")
        if len(df_cleaned) == 0:
            raise ValueError("No data left after cleaning. Check data_path and columns.")
    except Exception as e:
        logger.error(f"FATAL: Failed to load data. Error: {e}")
        return
    logger.info("-" * 50)

    # --- 3. Initializing and Running ExpRunner ---
    logger.info("--- 3. Initializing ExpRunner & Running 10% Experiment ---")
    try:
        runner = ExpRunner.create(
            exp_params=exp_config,
            full_df=df_cleaned,
            emb_params=emb_params,
            feat_proc_params=feat_params,
            model_params=mlp_params,
            field_config=field_config
        )

        # This one call runs the entire pipeline:
        # split -> 10% frac -> embed -> process -> train -> evaluate
        results = runner.run_training_set_size(fractions=[0.1])

        logger.info("--- Dry Run Experiment Results (10% frac) ---")
        logger.info(results)

    except Exception as e:
        logger.error(f"FATAL: ExpRunner failed during execution. Error: {e}", exc_info=True)
        return
    logger.info("-" * 50)

    logger.info("=" * 50)
    logger.info("--- DRY RUN SUCCESSFUL ---")
    logger.info("All components ran end-to-end.")
    logger.info("=" * 50)


if __name__ == "__main__":
    setup_logging(Path('logs/'), "dry_run")
    logger = logging.getLogger(__name__)
    main()