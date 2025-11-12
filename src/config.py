from dataclasses import dataclass
from enum import StrEnum
import numpy as np
import torch


TWO_PI = 2 * np.pi


@dataclass(frozen=True)
class ExperimentConfig:
    test_size:float = 0.2
    random_state:int = 2025
    max_text_length:int = 64 # Max tokens for the text feature
    go_no_go_threshold_pct:float = 5.0 # Max 5% unknown tokens

@dataclass(frozen=True)
class FieldConfig:
    """A single object to hold all column name constants."""
    date: str = 'date'
    amount: str = 'amount'
    text: str = 'bankRawDescription'
    label: str = 'isRecurring'
    accountId: str = 'accountId'
    trId: str = 'id' # transactionId

class EmbModel(StrEnum):
    """Using StrEnum for our model name constants."""
    ALBERT = "albert-base-v2"
    DISTILBERT = "distilbert-base-uncased"
    FINBERT = "ProsusAI/finbert"
    MiniLM_L12 = 'sentence-transformers/all-MiniLM-L12-v2'


@dataclass(frozen=True)
class FilterConfig:
    """
    Holds all the "tunable" hyperparameters for our Stage 2 Filter.
    These are the values you would "train" using a Hyperparameter Optimizer.
    """
    # DBSCAN parameters
    dbscan_eps: float = 0.5       # The "distance" to consider neighbors. HIGHLY TUNABLE.
    dbscan_min_samples: int = 2   # The minimum transactions to form a "group".

    # Filter parameters
    min_txns_for_period: int = 3  # We need 3+ txns to find a stable period.
    amount_std_threshold: float = 1.0  # Max $1.00 stdev in amount
    date_std_threshold: float = 2.0    # Max 2.0 day stdev in time deltas


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    # if torch.xpu.is_available():
    #     return torch.device("xpu")
    return torch.device("cpu")
