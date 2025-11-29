import argparse
import logging
import pickle
import torch
import pandas as pd
from pathlib import Path
from multi.config import MultiExpConfig, MultiFieldConfig
from multi.encoder import TransactionTransformer
from multi.tune_multi import analyze_mistakes

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("analyze_mistakes")

def load_pickle_data(pickle_path):
    logger.info(f"Loading data from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_model(model_path, config):
    logger.info(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=config.device)
    
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # If config is saved, we could use it, but we'll use the passed config for now 
        # (assuming it matches or we want to override)
    else:
        state_dict = checkpoint

    model = TransactionTransformer(config)
    model.load_state_dict(state_dict)
    model.to(config.device)
    return model

def main():
    parser = argparse.ArgumentParser(description="Analyze Model Mistakes")
    parser.add_argument("--pickle_path", type=str, required=True, help="Path to the split pickle file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Which split to analyze")
    parser.add_argument("--num_examples", type=int, default=10, help="Number of examples to show")
    
    args = parser.parse_args()

    # 1. Load Data
    data = load_pickle_data(args.pickle_path)
    if args.split not in data:
        raise ValueError(f"Split '{args.split}' not found in pickle file. Available: {list(data.keys())}")
    
    df = data[args.split]
    use_cp = data.get('use_counter_party', True)
    logger.info(f"Loaded {len(df)} rows for split '{args.split}'. use_counter_party={use_cp}")

    # 2. Config
    # We need to ensure the config matches what the model expects
    # Ideally we load it from the checkpoint, but for now we create a default and update key params
    config = MultiExpConfig()
    config.use_counter_party = use_cp
    
    # Try to load config from checkpoint if possible
    try:
        checkpoint = torch.load(args.model_path, map_location='cpu')
        if "config" in checkpoint:
            saved_config = checkpoint["config"]
            # Update critical runtime params
            config = saved_config
            logger.info("Loaded config from checkpoint.")
    except Exception as e:
        logger.warning(f"Could not load config from checkpoint: {e}. Using default.")

    # 3. Load Model
    model = load_model(args.model_path, config)

    # 4. Run Analysis
    analyze_mistakes(model, df, config, num_examples=args.num_examples)

if __name__ == "__main__":
    main()
