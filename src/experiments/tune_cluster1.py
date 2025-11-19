from pathlib import Path
from clustering_exp import ClusteringExperiment
from log_utils import setup_logging
from config import EmbModel

setup_logging(Path('logs/'), "tune_compare")

grid = {
    # --- Compare High Level Strategies ---
    'strategy': ['greedy', 'dbscan'],

    # --- Compare Math (MAD vs STD) ---
    'stability_metric': ['std', 'mad'],

    # --- Shared Constraints ---
    'date_var': [2.0],
    'amt_var': [1.0],

    # --- Greedy Params (Only affects greedy runs) ---
    'greedy_sim': [0.90, 0.92],

    # --- DBSCAN Params (Only affects dbscan runs) ---
    'dbscan_eps': [0.3, 0.5]
}

if __name__ == "__main__":
    tuner = ClusteringExperiment(experiment_name="v1_comparison", filter_direction=1)
    tuner.load_data()
    tuner.precompute_embeddings(EmbModel.MPNET)
    tuner.run_grid(grid)