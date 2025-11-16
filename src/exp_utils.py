import torch
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

def set_global_seed(seed: int):
    """
    Sets the global random seed for PyTorch, NumPy, and Python's random module.
    """
    logger.info(f"Setting global random seed to {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # These two lines are sometimes needed for 100% determinism
    # but can slow down training. We can add them if we still see variance.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False