import argparse
import pandas as pd
import torch
import random
import numpy as np
from multi.config import MultiExpConfig
from multi.data import get_dataloader
from multi.encoder import TransactionTransformer
from multi.trainer import MultiTrainer


def mock_data_generator(num_accounts=100):
    """
    Creates dummy data if no CSV is found.
    """
    print("Generating synthetic data for testing...")
    data = []

    for acc_id in range(num_accounts):
        num_tx = random.randint(20, 150)
        for i in range(num_tx):
            is_rec = random.random() < 0.2
            cycle = random.choice(['monthly', 'onceAWeek']) if is_rec else 'None'
            pat_id = random.randint(1, 5) if is_rec else None

            # Randomize amount sign for credit/debit testing
            amount = 15.99 if is_rec else random.uniform(5, 100)
            if random.random() > 0.8: amount *= -1  # 20% credits

            data.append({
                'accountId': acc_id,
                'transactionId': f"{acc_id}_{i}",
                'bankName': 'MockBank',
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
    args = parser.parse_args()

    # Setup
    config = MultiExpConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load Data
    if args.data:
        df = pd.read_csv(args.data)
    else:
        df = mock_data_generator()

    print(f"Loaded {len(df)} transactions.")

    # Split
    account_ids = df['accountId'].unique()
    np.random.shuffle(account_ids)
    split = int(len(account_ids) * 0.8)
    train_ids, val_ids = account_ids[:split], account_ids[split:]

    train_df = df[df['accountId'].isin(train_ids)]
    val_df = df[df['accountId'].isin(val_ids)]

    train_loader = get_dataloader(train_df, config, shuffle=True)
    val_loader = get_dataloader(val_df, config, shuffle=False)

    # Model
    print(f"Initializing model on {config.device}...")
    model = TransactionTransformer(config)

    # Train
    trainer = MultiTrainer(model, config)

    for epoch in range(config.num_epochs):
        train_loss = trainer.train_epoch(train_loader, epoch + 1)
        metrics = trainer.evaluate(val_loader)

        print(f"Epoch {epoch + 1} | Loss: {train_loss:.4f} | Val F1 (Pairwise): {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()