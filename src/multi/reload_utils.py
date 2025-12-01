import logging
import pickle
import torch
from pathlib import Path
from multi.config import MultiExpConfig
from multi.encoder import TransactionTransformer
from multi.tune_multi import get_data_cache_path

logger = logging.getLogger(__name__)


def resolve_device(device_arg: str | None = "auto") -> torch.device:
    """
    Resolves the device based on argument and availability.
    """
    if device_arg is None or device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model_for_eval(model_path: str | Path, device_str: str = "auto") -> tuple[
    TransactionTransformer, MultiExpConfig, torch.device]:
    """
    Loads a checkpoint, initializes the model architecture, loads weights,
    and sets the model to evaluation mode on the specified device.

    Returns:
        model: The initialized TransactionTransformer (eval mode).
        config: The MultiExpConfig loaded from the checkpoint.
        device: The torch.device the model is placed on.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")

    logger.info(f"Loading checkpoint from {path}...")
    # Load to CPU first to avoid CUDA OOM if device maps change
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if 'config' not in checkpoint:
        raise ValueError("Checkpoint dictionary does not contain 'config' key.")

    config: MultiExpConfig = checkpoint['config']

    # Resolve device
    device = resolve_device(device_str)
    logger.info(f"Setting up model on device: {device}")

    # Initialize and load
    model = TransactionTransformer(config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    return model, config, device


def load_data_for_config(config: MultiExpConfig) -> tuple:
    """
    Loads the cached train/val/test splits that correspond to the
    random_state and downsample settings in the config.
    """
    cache_path = get_data_cache_path(
        random_state=config.random_state,
        downsample=config.downsample
    )

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cached data not found at {cache_path}. "
            f"Expected cache for seed={config.random_state}, downsample={config.downsample}. "
            "Please ensure training/data prep was run."
        )

    logger.info(f"Loading cached dataset splits from {cache_path}...")
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)

    # Returns train_df, val_df, test_df
    return data['train'], data['val'], data['test']