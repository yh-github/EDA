import argparse
import logging
import pickle
from pathlib import Path
import torch
from multi.encoder import TransactionTransformer
from multi.analysis import analyze_classification_mistakes, analyze_adjacency_mistakes

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("analyze_mistakes")


def load_cached_data(cache_dir: Path):
    """
    Finds the most recent split pickle in the cache directory.
    """
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory {cache_dir} does not exist.")

    files = list(cache_dir.glob("split_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No split_*.pkl files found in {cache_dir}")

    # Sort by modification time (newest first)
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_file = files[0]

    logger.info(f"Loading data from cached split: {latest_file}")
    with open(latest_file, 'rb') as f:
        data = pickle.load(f)

    return data['train'], data['val'], data['test']


def main():
    parser = argparse.ArgumentParser(description="Analyze mistakes of a trained Multi model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .pth checkpoint file.")
    parser.add_argument("--cache_dir", type=str, default="cache/data", help="Directory containing split_*.pkl")
    parser.add_argument("--device", type=str, default="auto", help="cpu or cuda")
    parser.add_argument("--examples", type=int, default=10, help="Number of examples to show per error type")

    args = parser.parse_args()

    # 1. Load Model & Config
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return

    logger.info(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # Load to CPU first

    if 'config' not in checkpoint:
        logger.error("Checkpoint does not contain 'config' key.")
        return

    config = checkpoint['config']

    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    config.device = device  # Update config with current runtime device
    logger.info(f"Running on {device}")

    # Initialize Model
    model = TransactionTransformer(config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # 2. Load Data (Assumed saved)
    try:
        _, val_df, test_df = load_cached_data(Path(args.cache_dir))
        logger.info(f"Data Loaded. Val: {len(val_df)}, Test: {len(test_df)}")
    except Exception as e:
        logger.error(f"Failed to load cached data: {e}")
        return

    # 3. Run Analysis
    logger.info("=" * 60)
    logger.info(" Analyzing VALIDATION Set Mistakes")
    logger.info("=" * 60)
    analyze_classification_mistakes(model, val_df, config, num_examples=args.examples)
    analyze_adjacency_mistakes(model, val_df, config, num_examples=args.examples)

    logger.info("\n" + "=" * 60)
    logger.info(" Analyzing TEST Set Mistakes")
    logger.info("=" * 60)
    analyze_classification_mistakes(model, test_df, config, num_examples=args.examples)
    analyze_adjacency_mistakes(model, test_df, config, num_examples=args.examples)


if __name__ == "__main__":
    main()