import os

from common.exp_utils import set_global_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import logging
import pandas as pd
import torch
import random
import numpy as np
from pathlib import Path
from multi.config import MultiExpConfig, MultiFieldConfig
from multi.data import get_dataloader
from multi.encoder import TransactionTransformer
from multi.trainer import MultiTrainer
from common.data import create_train_val_test_split

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mock_data_generator(num_accounts=100):
    """Creates dummy data if no CSV is found."""
    logger.info("Generating synthetic data for testing...")
    data = []
    for acc_id in range(num_accounts):
        num_tx = random.randint(20, 150)
        for i in range(num_tx):
            is_rec = random.random() < 0.2
            cycle = random.choice(['monthly', 'onceAWeek']) if is_rec else 'None'
            pat_id = random.randint(1, 5) if is_rec else None
            amount = 15.99 if is_rec else random.uniform(5, 100)
            if random.random() > 0.5:
                amount *= -1

            data.append({
                'accountId': acc_id,
                'transactionId': f"{acc_id}_{i}",
                'bankRawDescription': f"PAYMENT TO {'NETFLIX' if is_rec else 'STORE'} {random.randint(1, 100)}",
                'counterParty': 'Netflix' if is_rec else '',  # Include CP column for mock
                'amount': amount,
                'date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=random.randint(0, 150)),
                'isRecurring': is_rec,
                'patternId': pat_id,
                'patternCycle': cycle
            })
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description="Multi Bank Transaction Pattern Detector")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--data", type=str, default="data/all_data.csv", help="Path to CSV data")
    parser.add_argument("--output_dir", type=str, default="checkpoints/multi", help="Dir to save model")
    parser.add_argument("--downsample", type=float, default=0.3, help="Fraction of accounts to use (0.0-1.0)")
    args = parser.parse_args()

    # 1. Config & Setup
    config = MultiExpConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.output_dir = args.output_dir

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    set_global_seed(config.random_state)

    # 2. Load Data
    if args.data and args.data.lower() == "mock":
        df = mock_data_generator()
    elif args.data and os.path.exists(args.data):
        df = pd.read_csv(args.data)
    else:
        raise Exception(f"No CSV found path={args.data}")

    logger.info(f"Loaded {len(df)} transactions.")
    field_config = MultiFieldConfig()
    df[field_config.accountId] = df[field_config.accountId].astype(str)
    df[field_config.trId] = df[field_config.trId].astype(str)

    # 3. Feature Availability Check
    if field_config.counter_party in df.columns:
        # Check coverage (non-empty)
        # We treat empty strings and NaNs as "missing"
        valid_cp = df[field_config.counter_party].replace('', np.nan).notna()
        coverage = valid_cp.sum() / len(df)
        logger.info(f"Counter Party Coverage: {coverage:.2%}")

        if coverage < 0.5:
            logger.warning("Counter Party coverage low (<50%). Disabling feature.")
            config.use_counter_party = False
        else:
            logger.info("Counter Party coverage good. Enabling feature.")
            config.use_counter_party = True
    else:
        logger.warning("Counter Party column missing. Disabling feature.")
        config.use_counter_party = False

    # 4. Downsample (Account-based)
    if 0.0 < args.downsample < 1.0:
        logger.info(f"Downsampling to {args.downsample:.0%} of accounts...")
        account_ids = df[field_config.accountId].unique()
        rng = np.random.default_rng(config.random_state)
        n_select = max(1, int(len(account_ids) * args.downsample))
        selected_ids = rng.choice(account_ids, size=n_select, replace=False)
        df = df[df[field_config.accountId].isin(selected_ids)].copy()
        logger.info(f"Dataset size after downsampling: {len(df)} rows ({len(selected_ids)} accounts)")

    # 5. Split Data
    logger.info("Splitting data into Train/Val/Test...")
    train_df, val_df, test_df = create_train_val_test_split(
        test_size=0.1,
        val_size=0.1,
        full_df=df,
        random_state=config.random_state,
        field_config=field_config
    )

    # 6. Data Loaders
    train_loader = get_dataloader(train_df, config, shuffle=True)
    val_loader = get_dataloader(val_df, config, shuffle=False)

    # 7. Model Initialization
    logger.info(f"Initializing model on {config.device} (use_cp={config.use_counter_party})...")
    model = TransactionTransformer(config)
    trainer = MultiTrainer(model, config, pos_weight=2.5)

    # 8. Training Loop
    best_f1 = -1.0
    save_path = os.path.join(config.output_dir, "model.pth")

    for epoch in range(config.num_epochs):
        train_loss = trainer.train_epoch(train_loader, epoch + 1)
        metrics = trainer.evaluate(val_loader)

        logger.info(f"Epoch {epoch + 1}/{config.num_epochs} | Loss: {train_loss:.4f} | Val F1: {metrics['f1']:.4f}")

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            checkpoint = {
                "config": config,
                "state_dict": model.state_dict()
            }
            torch.save(checkpoint, save_path)
            logger.info(f"  --> New Best Model Saved to {save_path}")

    logger.info("Training Complete.")


if __name__ == "__main__":
    main()