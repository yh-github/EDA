import os
from datetime import datetime
from common.exp_utils import set_global_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import logging
import pandas as pd
import torch
import random
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from multi.config import MultiExpConfig, MultiFieldConfig
from multi.data import get_dataloader, analyze_token_distribution
from multi.encoder import TransactionTransformer
from multi.trainer import MultiTrainer
from common.data import create_train_val_test_split
from common.log_utils import flush_logger

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
                'counter_party': 'Netflix' if is_rec else '',
                'amount': amount,
                'date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=random.randint(0, 150)),
                'isRecurring': is_rec,
                'patternId': pat_id,
                'patternCycle': cycle
            })
    return pd.DataFrame(data)


def main():
    config = MultiExpConfig()

    parser = argparse.ArgumentParser(description="Multi Bank Transaction Pattern Detector")
    parser.add_argument("--epochs", type=int, default=config.num_epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("--data_path", type=str, default=config.data_path, help="Path to CSV data")
    parser.add_argument("--output_dir", type=str, default=config.output_dir, help="Dir to save model")
    parser.add_argument("--downsample", type=float, default=config.downsample,
                        help="Fraction of accounts to use (0.0-1.0)")
    parser.add_argument("--accumulate", type=int, default=config.gradient_accumulation_steps,
                        help="Gradient accumulation steps")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    args = parser.parse_args()

    # 1. Config & Setup
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.output_dir = args.output_dir
    config.data_path = args.data_path
    config.downsample = args.downsample
    config.gradient_accumulation_steps = args.accumulate

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    set_global_seed(config.random_state)

    # 2. Load Data
    if args.data and args.data.lower() == "mock":
        df = mock_data_generator()
    elif args.data and os.path.exists(args.data):
        df = pd.read_csv(args.data, low_memory=False)
    else:
        raise Exception(f"No CSV found path={args.data}")

    logger.info(f"Loaded {len(df)} transactions.")
    field_config = MultiFieldConfig()
    df[field_config.accountId] = df[field_config.accountId].astype(str)
    df[field_config.trId] = df[field_config.trId].astype(str)

    # 3. Feature Availability Check
    if field_config.counter_party in df.columns:
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
    if 0.0 < config.downsample < 1.0:
        logger.info(f"Downsampling to {config.downsample:.0%} of accounts...")
        account_ids = df[field_config.accountId].unique()
        rng = np.random.default_rng(config.random_state)
        n_select = max(1, int(len(account_ids) * config.downsample))
        selected_ids = rng.choice(account_ids, size=n_select, replace=False)
        df = df[df[field_config.accountId].isin(selected_ids)].copy()
        logger.info(f"Dataset size after downsampling: {len(df)} rows ({len(selected_ids)} accounts)")

    # --- 4b. TOKEN STATS REPORT ---
    logger.info("Running Token Stats Analysis...")
    tokenizer_for_stats = AutoTokenizer.from_pretrained(config.text_encoder_model)
    analyze_token_distribution(df, tokenizer_for_stats, config)
    # Ensure logs are printed before potentially heavy processing
    flush_logger()

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

    # Optional: Compile
    if args.compile:
        try:
            logger.info("Compiling model with torch.compile...")
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Proceeding without compilation.")

    trainer = MultiTrainer(model, config, pos_weight=2.5)

    # 8. Training Loop
    best_f1 = -1.0
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(config.output_dir, f"model_{timestamp}.pth")

    # Early Stopping Variables
    patience = config.early_stopping_patience
    patience_counter = 0

    for epoch in range(config.num_epochs):
        train_loss = trainer.train_epoch(train_loader, epoch + 1)
        metrics = trainer.evaluate(val_loader)

        val_f1 = metrics['f1']
        val_loss = metrics['val_loss']

        # Improved Logging
        logger.info(
            f"Epoch {epoch + 1}/{config.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Prec: {metrics['precision']:.4f} | "
            f"Rec: {metrics['recall']:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0  # Reset patience

            checkpoint = {
                "config": config,
                "state_dict": model.state_dict(),
                # Note: if compiled, might need ._orig_mod.state_dict() in some versions
                "best_f1": best_f1,
                "epoch": epoch + 1
            }
            # Handle compiled model saving
            if args.compile and hasattr(model, '_orig_mod'):
                checkpoint["state_dict"] = model._orig_mod.state_dict()

            torch.save(checkpoint, save_path)
            logger.info(f"  --> New Best Model Saved (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  ... No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                logger.info(f"⛔ Early stopping triggered at epoch {epoch + 1}")
                break

    logger.info("Training Complete.")


if __name__ == "__main__":
    main()