import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import all your project modules
from config import (
    ExperimentConfig,
    FieldConfig,
    EmbModel,
    get_device
)
from embedder import EmbeddingService
from feature_processor import FeatProcParams, FeatureHyperParams
from runner import ExpRunner
from classifier import HybridModel
from data import TransactionDataset, TrainingSample

# --- Setup Logging (copied from your notebook) ---
stdout_handler = logging.StreamHandler(sys.stdout)
def setup_logging(log_dir:Path):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"DRY_RUN_{timestamp}.log"

    file_handler = logging.FileHandler(log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[stdout_handler, file_handler]
    )

setup_logging(Path('../logs/dry_run'))
logger = logging.getLogger(__name__)
# --------------------------------------------------


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

    # --- 1. System Info ---
    logger.info("--- 1. System & Device ---")
    device = get_device()
    logger.info(f"PyTorch device available: {device}")
    logger.info(f"Torch version: {torch.__version__}")
    logger.info("-" * 50)

    # --- 2. Config Initialization ---
    logger.info("--- 2. Initializing Configs ---")
    # Use hardcoded configs for a simple test
    exp_config = ExperimentConfig()
    field_config = FieldConfig()
    # Use feature-rich params to test the full pipeline
    feat_params = FeatProcParams(n_bins=20, k_top=50) 
    emb_params = EmbeddingService.Params(model_name=EmbModel.ALBERT)
    
    logger.info(f"ExperimentConfig: {exp_config}")
    logger.info(f"FeatProcParams: {feat_params}")
    logger.info(f"EmbModel: {emb_params.model_name}")
    logger.info("-" * 50)

    # --- 3. Data Loading ---
    logger.info(f"--- 3. Loading Data from {args.data_path} ---")
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

    # --- 4. Runner & Embedder ---
    logger.info("--- 4. Initializing ExpRunner & Embedder ---")
    try:
        runner = ExpRunner.create(
            exp_params=exp_config,
            full_df=df_cleaned,
            emb_params=emb_params,
            feat_proc_params=feat_params,
            field_config=field_config
        )
        # This call will load the model from cache or disk
        embedder = runner.get_embedder(emb_params)
        logger.info(f"Embedder '{embedder.model_name}' loaded successfully.")
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize ExpRunner or Embedder. Error: {e}")
        return
    logger.info("-" * 50)

    # --- 5. Data Pipeline Test ---
    logger.info("--- 5. Testing Data Split & Feature Generation (on 10% frac) ---")
    try:
        df_train, df_test = runner.split_data_by_group()
        logger.info(f"Split data: {len(df_train)} train rows, {len(df_test)} test rows.")
        
        # Get just the first, smallest fraction to test
        test_frac = 0.1
        frac, sub_train_df = next(runner.create_learning_curve_splits(df_train, [test_frac]))
        logger.info(f"Using {frac*100}% fraction: {len(sub_train_df)} rows.")
        
        # This is the core test: embeds text, fits processor, transforms data
        train_fs, test_fs, processor, meta = runner.build_data_for_pytorch(sub_train_df, df_test)
        
        logger.info("Feature generation successful. Shapes:")
        logger.info(f"  Train Text:       {train_fs.X_text.shape}")
        logger.info(f"  Train Continuous: {train_fs.X_continuous.shape}")
        logger.info(f"  Train Categorical:  {train_fs.X_categorical.shape}")
        logger.info(f"  Train Labels:     {train_fs.y.shape}")
        
        logger.info(f"Generated Metadata: {meta}")
    except Exception as e:
        logger.error(f"FATAL: Failed during data processing pipeline. Error: {e}", exc_info=True)
        return
    logger.info("-" * 50)

    # --- 6. Model Instantiation Test ---
    logger.info("--- 6. Testing Model Instantiation ---")
    try:
        feature_config = FeatureHyperParams.build(train_fs, meta)
        mlp_config = HybridModel.MlpHyperParams()
        logger.info(f"Built FeatureHyperParams: {feature_config}")
        logger.info(f"Built MlpHyperParams: {mlp_config}")
        
        model = HybridModel(
            feature_config,
            mlp_config
        )
        logger.info("Model instantiated successfully.")
        print("\n" + str(model) + "\n") # Print the model architecture
    except Exception as e:
        logger.error(f"FATAL: Failed to instantiate model. Check config. Error: {e}", exc_info=True)
        return
    logger.info("-" * 50)
    
    # --- 7. DataLoader Test ---
    logger.info("--- 7. Testing DataLoader ---")
    try:
        train_dataset = TransactionDataset(train_fs)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=4, # Small batch for testing
            shuffle=True, 
            collate_fn=TrainingSample.collate_fn
        )
        
        # Get one batch from the loader
        batch = next(iter(train_loader))
        logger.info("Successfully pulled one batch from DataLoader.")
        logger.info(f"  Batch Text Shape:       {batch.x_text.shape}")
        logger.info(f"  Batch Continuous Shape: {batch.x_continuous.shape}")
        logger.info(f"  Batch Categorical Shape:  {batch.x_categorical.shape}")
        logger.info(f"  Batch Labels Shape:     {batch.y.shape}")
        
    except Exception as e:
        logger.error(f"FATAL: Failed to get batch from DataLoader. Error: {e}", exc_info=True)
        return
    logger.info("-" * 50)

    logger.info("=" * 50)
    logger.info("--- DRY RUN SUCCESSFUL ---")
    logger.info("All components initialized and connected.")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
