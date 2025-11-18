from pathlib import Path
from log_utils import setup_logging
from greedy_tuner import GreedyTuner
from config import EmbModel

# Setup
setup_logging(Path('logs/'), "tune_greedy")

# --- The Grid ---
# We can tune the semantic threshold and the logic constraints separately
grid_search_config = {
    # 1. Semantic Strictness (The big lever)
    # Lower = More candidates, Higher = Fewer False Positives
    'sim_threshold': [0.85, 0.90, 0.92, 0.95],

    # 2. Amount Logic
    'amount_tol_abs': [1.0, 2.0, 5.0],  # Allow $1, $2, or $5 wiggle
    'amount_tol_pct': [0.0, 0.05, 0.10],  # Allow 0%, 5%, or 10% wiggle

    # 3. Time Logic
    'date_std_threshold': [1.0, 2.0, 3.0],  # Allow 1, 2, or 3 days jitter
    'min_txns': [2, 3]  # Need 2 or 3 to confirm?
}

if __name__ == "__main__":
    tuner = GreedyTuner(ind=1, filter_direction=1)
    tuner.load_data()

    # Pre-compute MPNET embeddings so the loop is fast
    tuner.precompute_embeddings(model_name=EmbModel.MPNET)

    # Run
    tuner.run_grid(grid_search_config)