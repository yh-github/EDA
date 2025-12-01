import argparse
import logging
from multi.analysis import analyze_classification_mistakes, analyze_adjacency_mistakes
from multi.data import get_dataloader
from multi.reload_utils import load_model_for_eval, load_data_for_config

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("analyze_mistakes")


def main():
    parser = argparse.ArgumentParser(description="Analyze mistakes of a trained Multi model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .pth checkpoint file.")
    # data_cache argument removed as it is inferred from config via reload_utils,
    # but kept in parser for backward compatibility if needed, though ignored.
    parser.add_argument("--data_cache", type=str, default="cache/data", help="(Ignored) Directory containing split_*.pkl")
    parser.add_argument("--device", type=str, default="auto", help="cpu or cuda")
    parser.add_argument("--examples", type=int, default=10, help="Number of examples to show per error type")

    args = parser.parse_args()

    # 1. Load Model & Config using Utils
    try:
        model, config, device = load_model_for_eval(args.model_path, args.device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    logger.info(f"Model loaded. Configured device: {device}")

    # 2. Load Data (Assumed saved)
    try:
        _, val_df, test_df = load_data_for_config(config)
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
    # The analysis functions in `multi.analysis` have been updated to use model.device
    logger.info("=" * 60)
    logger.info(" Analyzing VALIDATION Set Mistakes")
    logger.info("=" * 60)
    analyze_classification_mistakes(model, val_df, val_loader, config, num_examples=args.examples)
    analyze_adjacency_mistakes(model, val_df, val_loader, config, num_examples=args.examples)

    logger.info("\n" + "=" * 60)
    logger.info(" Analyzing TEST Set Mistakes")
    logger.info("=" * 60)
    analyze_classification_mistakes(model, test_df, test_loader, config, num_examples=args.examples)
    analyze_adjacency_mistakes(model, test_df, test_loader, config, num_examples=args.examples)


if __name__ == "__main__":
    main()