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

def load_model_and_config(model_path, device_override=None):
    logger.info(f"Loading model from {model_path}...")
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Checkpoint format not recognized (expected dict with 'state_dict')")

    # 1. Recover Config
    if "config" in checkpoint:
        config = checkpoint["config"]
        logger.info("✅ Loaded config from checkpoint.")
    else:
        logger.warning("⚠️ Config not found in checkpoint! Using default config (risky).")
        config = MultiExpConfig()

    # Override device if needed
    if device_override:
        # We can't easily change the config object if it's frozen, but we can use the device for moving the model
        device = device_override
    else:
        device = config.device

    # 2. Initialize Model
    model = TransactionTransformer(config)
    
    # 3. Load Weights
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    
    return model, config

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

    # 2. Load Model & Config
    # We load the model first to get the correct config used during training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model_and_config(args.model_path, device_override=device)

    # 3. Override Data Params if needed (e.g. use_counter_party from data split)
    # The model architecture is fixed by the loaded config, but we need to ensure
    # the data processing matches what the model expects.
    # Ideally, the saved config matches the data, but if we forced CP off in data,
    # we should check if model expects it.
    
    if config.use_counter_party != use_cp:
        logger.warning(f"⚠️ Config mismatch! Model expects use_counter_party={config.use_counter_party}, but data has {use_cp}.")
        # If model expects CP but data doesn't have it, we might crash.
        # If model doesn't expect CP but data has it, we just ignore data.
        
    # 4. Run Analysis
    analyze_mistakes(model, df, config, num_examples=args.num_examples)

if __name__ == "__main__":
    main()
