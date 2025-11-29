import torch
import pandas as pd
import logging
import shutil
from pathlib import Path
from multi.config import MultiExpConfig, MultiFieldConfig
from multi.data import get_dataloader
from multi.encoder import TransactionTransformer
from multi.trainer import MultiTrainer
from common.data import create_mock_data, create_mock_account_data

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_local")

def test_training_loop():
    logger.info("Starting local training loop verification...")

    # 1. Create Mock Data
    field_config = MultiFieldConfig()
    # Create enough data for a few batches
    df = create_mock_data(random_state=42, n_samples=100)
    
    # Ensure we have the necessary columns for multi-model
    # The mock data creator might not add 'accountId' or 'patternId' exactly as needed for multi
    if field_config.accountId not in df.columns:
        df[field_config.accountId] = "ACC_1"
    if field_config.patternId not in df.columns:
        df[field_config.patternId] = -1
    if field_config.patternCycle not in df.columns:
        df[field_config.patternCycle] = "None"
    
    # Add a counter party column if missing
    if field_config.counter_party not in df.columns:
        df[field_config.counter_party] = "Mock CP"

    # 2. Config
    config = MultiExpConfig(
        num_epochs=2,
        batch_size=4,
        hidden_dim=32, # Small dim for speed
        num_heads=2,
        num_layers=1,
        use_focal_loss=True,
        scheduler_type='cosine',
        scheduler_t0=2,
        data_path="mock_data", # Not used directly here
        output_dir="test_output"
    )

    # 3. DataLoader
    logger.info("Creating DataLoader...")
    loader = get_dataloader(df, config, shuffle=True, n_workers=0)

    # 4. Model
    logger.info("Initializing Model...")
    model = TransactionTransformer(config)

    # 5. Trainer
    logger.info("Initializing Trainer...")
    trainer = MultiTrainer(model, config)

    # 6. Run Fit
    logger.info("Running Fit...")
    try:
        score = trainer.fit(
            train_loader=loader,
            val_loader=loader, # Use same for test
            epochs=config.num_epochs,
            metric_to_track='pr_auc'
        )
        logger.info(f"Training finished successfully. Score: {score}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise e
    finally:
        # Cleanup
        if Path("test_output").exists():
            shutil.rmtree("test_output")

if __name__ == "__main__":
    test_training_loop()
