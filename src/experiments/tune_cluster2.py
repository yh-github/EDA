from pathlib import Path

from clustering_exp import ClusteringExperiment
from log_utils import setup_logging
from config import EmbModel

# Log to a new file
setup_logging(Path('logs/'), "tune_lexical")

grid = {
    'strategy': ['lexical'],

    # String similarity thresholds (0.0 to 1.0)
    # 0.7 might catch "Netflix" vs "Netflix.com", 0.9 is strict
    'lexical_sim': [0.6, 0.7, 0.8, 0.9],

    # Stability constraints
    'stability_metric': ['mad'],  # Robust is likely better
    'date_var': [2.0],
    'amt_var': [1.0]
}

if __name__ == "__main__":
    # experiment_name is used for caching
    tuner = ClusteringExperiment(experiment_name="lexical_v1", filter_direction=1)
    tuner.load_data()
    # Note: We technically don't need embeddings for purely lexical,
    # but the loader currently expects them. We can pass ALBERT/MPNET.
    tuner.precompute_embeddings(EmbModel.MPNET)

    tuner.run_grid(grid)