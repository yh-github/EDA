import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datetime import datetime
from common.exp_utils import set_global_seed
import argparse
import logging
from pathlib import Path
from transformers import AutoTokenizer
from multi.config import MultiExpConfig, MultiFieldConfig
from multi.data import get_dataloader, analyze_token_distribution
from multi.encoder import TransactionTransformer
from multi.trainer import MultiTrainer
from multi.data_utils import load_and_prepare_data
from common.data import create_train_val_test_split
from common.log_utils import flush_logger

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    config = MultiExpConfig()

    parser = argparse.ArgumentParser(description="Multi Bank Transaction Pattern Detector")
    parser.add_argument("--num_epochs", type=int, default=config.num_epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config.gradient_accumulation_steps, help="gradient_accumulation_steps")
    parser.add_argument("--data_path", type=str, default=config.data_path, help="Path to CSV data")
    parser.add_argument("--output_dir", type=str, default=config.output_dir, help="Dir to save model")
    parser.add_argument("--downsample", type=float, default=config.downsample,
                        help="Fraction of accounts to use (0.0-1.0)")
    args = parser.parse_args()

    # 1. Config & Setup
    config = MultiExpConfig(**vars(args))

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    set_global_seed(config.random_state)

    # 2. Load and Prepare Data (Shared Utility)
    df = load_and_prepare_data(config)

    # 3. Token Stats Report (Run specific)
    logger.info("Running Token Stats Analysis...")
    tokenizer_for_stats = AutoTokenizer.from_pretrained(config.text_encoder_model)
    analyze_token_distribution(df, tokenizer_for_stats, config)
    flush_logger()

    # 4. Split Data
    logger.info("Splitting data into Train/Val/Test...")
    train_df, val_df, test_df = create_train_val_test_split(
        test_size=0.1,
        val_size=0.1,
        full_df=df,
        random_state=config.random_state,
        field_config=MultiFieldConfig()
    )

    # 5. Data Loaders
    train_loader = get_dataloader(train_df, config, shuffle=True)
    val_loader = get_dataloader(val_df, config, shuffle=False)

    # 6. Model Initialization
    logger.info(f"Initializing model on {config.device} (use_cp={config.use_counter_party})...")
    model = TransactionTransformer(config)

    trainer = MultiTrainer(model, config, pos_weight=2.5)

    # 7. Training Loop using simplified `fit`
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(config.output_dir, f"model_{timestamp}.pth")

    best_f1 = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.num_epochs,
        save_path=save_path
    )

    logger.info(f"Training Complete. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()