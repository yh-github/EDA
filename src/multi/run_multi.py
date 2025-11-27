import argparse
import logging
import os
import pandas as pd
import torch
import random
import numpy as np
from pathlib import Path
from multi.config import MultiExpConfig
from multi.data import get_dataloader
from multi.encoder import TransactionTransformer
from multi.trainer import MultiTrainer

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
            if random.random() > 0.8: amount *= -1

            data.append({
                'accountId': acc_id,
                'transactionId': f"{acc_id}_{i}",
                'bankRawDescription': f"PAYMENT TO {'NETFLIX' if is_rec else 'STORE'} {random.randint(1, 100)}",
                'counter_party': 'Netflix' if is_rec else 'Unknown',
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
    parser.add_argument("--data", type=str, default=None, help="Path to CSV data")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Dir to save model")
    args = parser.parse_args()

    # 1. Config & Setup
    config = MultiExpConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.output_dir = args.output_dir

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 2. Load Data
    if args.data and os.path.exists(args.data):
        df = pd.read_csv(args.data)
    else:
        df = mock_data_generator()

    logger.info(f"Loaded {len(df)} transactions.")

    # 3. Split Data (Account-based)
    account_ids = df['accountId'].unique()
    np.random.shuffle(account_ids)
    split = int(len(account_ids) * 0.8)
    train_ids, val_ids = account_ids[:split], account_ids[split:]

    train_df = df[df['accountId'].isin(train_ids)].copy()
    val_df = df[df['accountId'].isin(val_ids)].copy()

    # 4. Data Loaders
    # Note: get_dataloader initializes the Tokenizer internally based on config
    train_loader = get_dataloader(train_df, config, shuffle=True)
    val_loader = get_dataloader(val_df, config, shuffle=False)

    # 5. Model Initialization
    logger.info(f"Initializing model on {config.device}...")
    model = TransactionTransformer(config)
    trainer = MultiTrainer(model, config)

    # 6. Training Loop
    best_f1 = -1.0
    save_path = os.path.join(config.output_dir, "model.pth")

    for epoch in range(config.num_epochs):
        train_loss = trainer.train_epoch(train_loader, epoch + 1)
        metrics = trainer.evaluate(val_loader)

        logger.info(f"Epoch {epoch + 1}/{config.num_epochs} | Loss: {train_loss:.4f} | Val F1: {metrics['f1']:.4f}")

        # Save Best Model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']

            checkpoint = {
                "config": config,  # Save the architecture params
                "state_dict": model.state_dict()  # Save the weights
            }

            torch.save(checkpoint, save_path)
            logger.info(f"  --> New Best Model Saved to {save_path}")

    logger.info("Training Complete.")


if __name__ == "__main__":
    main()