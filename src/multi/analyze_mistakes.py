import argparse
import logging
import pickle
from pathlib import Path
import torch

from multi.config import MultiExpConfig
from multi.encoder import TransactionTransformer
from multi.analysis import analyze_classification_mistakes, analyze_adjacency_mistakes
from multi.tune_multi import get_data_cache_path
from multi.data import get_dataloader

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("analyze_mistakes")


def load_cached_data(config: MultiExpConfig):
    """
    Finds the most recent split pickle in the cache directory.
    """
    data_cache_path = get_data_cache_path(
        random_state=config.random_state, downsample=config.downsample
    )

    if not data_cache_path.exists():
        raise FileNotFoundError(f"Cache directory {data_cache_path} does not exist.")

    with open(data_cache_path, 'rb') as f:
        data = pickle.load(f)

    return data['train'], data['val'], data['test']


def main():
    parser = argparse.ArgumentParser(description="Analyze mistakes of a trained Multi model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .pth checkpoint file.")
    parser.add_argument("--data_cache", type=str, default="cache/data", help="Directory containing split_*.pkl")
    parser.add_argument("--device", type=str, default="auto", help="cpu or cuda")
    parser.add_argument("--examples", type=int, default=10, help="Number of examples to show per error type")

    args = parser.parse_args()

    # 1. Load Model & Config
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return

    logger.info(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

    if 'config' not in checkpoint:
        logger.error("Checkpoint does not contain 'config' key.")
        return

    config: MultiExpConfig = checkpoint['config']

    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    config.device = device

    logger.info(f"Running on {device}")

    # Initialize Model
    model = TransactionTransformer(config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # 2. Load Data (Assumed saved)
    try:
        _, val_df, test_df = load_cached_data(config)
        logger.info(f"Data Loaded. Val: {len(val_df)}, Test: {len(test_df)}")
    except Exception as e:
        logger.error(f"Failed to load cached data: {e}")
        return

    # 3. Create Loaders (ONE TIME cost)
    logger.info("Initializing Validation Loader (Pre-tokenizing)...")
    val_loader = get_dataloader(val_df, config, shuffle=False, n_workers=0)

    logger.info("Initializing Test Loader (Pre-tokenizing)...")
    test_loader = get_dataloader(test_df, config, shuffle=False, n_workers=0)

    # 4. Run Analysis (Fast inference)
    logger.info("=" * 60)
    logger.info(" Analyzing VALIDATION Set Mistakes")
    logger.info("=" * 60)
    # Pass BOTH the dataframe AND the loader
    analyze_classification_mistakes(model, val_df, val_loader, config, num_examples=args.examples)
    analyze_adjacency_mistakes(model, val_df, val_loader, config, num_examples=args.examples)

    logger.info("\n" + "=" * 60)
    logger.info(" Analyzing TEST Set Mistakes")
    logger.info("=" * 60)
    analyze_classification_mistakes(model, test_df, test_loader, config, num_examples=args.examples)
    analyze_adjacency_mistakes(model, test_df, test_loader, config, num_examples=args.examples)


if __name__ == "__main__":
    main()