from typing import Any
import torch
import numpy as np
import random
import logging
import sys

logger = logging.getLogger(__name__)

def get_cli_args() -> dict[int, str]:
    return {i:v for i,v in enumerate(sys.argv)}

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

def exclude_none(data: Any) -> Any:
    """
    Recursively traverses a data structure (dict or list) and
    removes all keys from dictionaries that have a None value.
    """

    if data is None:
        return data

    if isinstance(data, dict):
        # Create a new dict, iterating and filtering
        new_dict = {}
        for k, v in data.items():
            if v is not None:
                # Recurse on the value
                new_dict[k] = exclude_none(v)
        return new_dict

    if isinstance(data, list):
        # Return a new list, recursing on each item
        return [exclude_none(item) for item in data]

    # Base case: return value as-is (int, str, bool, etc.)
    return data

import subprocess

def get_git_info():
    """
    Retrieves the current Git branch and the latest commit hash.
    Returns a tuple (branch_name, commit_hash) or (None, None) if not in a Git repo.
    """
    try:
        # Get current branch name
        branch_name = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.PIPE
        ).strip().decode('utf-8')

        # Get latest commit hash
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.PIPE
        ).strip().decode('utf-8')

        return f"{branch_name=} {commit_hash=}"
    except subprocess.CalledProcessError:
        raise Exception("Not in a Git repository.")