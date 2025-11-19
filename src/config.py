from dataclasses import dataclass
from enum import StrEnum
import numpy as np
import torch


TWO_PI = 2 * np.pi

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # if torch.xpu.is_available(): # XPU is buggy
    #     return torch.device("xpu")
    return torch.device("cpu")


@dataclass(frozen=True)
class ExperimentConfig:
    test_size:float = 0.2
    random_state:int = 112025
    max_text_length:int = 64 # Max tokens for the text feature
    go_no_go_threshold_pct:float = 5.0 # Max 5% unknown tokens

    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-3
    early_stopping_patience: int = 3  # Stop after 3 epochs of no improvement

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
    MPNET = 'sentence-transformers/all-mpnet-base-v2'


class ClusteringStrategy(StrEnum):
    GREEDY = "greedy"
    DBSCAN = "dbscan"
    HDBSCAN_ROBUST = "hdbscan_robust"  # The pattern_finder.py one


@dataclass(frozen=True)
class FilterConfig:
    # --- Master Switch ---
    strategy: ClusteringStrategy = ClusteringStrategy.GREEDY

    # --- Shared Constraints ---
    min_txns_for_period: int = 3

    # --- Stability Math (Shared) ---
    # Use 'std' (standard) or 'mad' (robust from pattern_finder)
    stability_metric: str = 'std'
    date_variance_threshold: float = 2.0  # Days
    amount_variance_threshold: float = 1.0  # Dollars

    # --- Strategy-Specific Params ---
    # Greedy
    greedy_sim_threshold: float = 0.90
    greedy_amount_tol_abs: float = 2.00
    greedy_amount_tol_pct: float = 0.05

    # DBSCAN
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 2