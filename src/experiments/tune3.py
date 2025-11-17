from hyper_tuner import HyperTuner
from config import EmbModel

# grid search around best from tune1

structured_grid_config = {
    'exp_params': {
        'learning_rate': [0.001],
        'batch_size': [64, 128]
    },
    'emb_params': {
        'model_name': [EmbModel.MiniLM_L12]  # Lock in winner
    },
    'feat_proc_params': {
        'use_cyclical_dates': [False, True],
        'use_categorical_dates': [False, True],
        'use_continuous_amount': [False, True],
        'use_categorical_amount': [True],
        'k_top': [10, 20, 30],
        'n_bins': [10, 20, 30]
    },
    'model_params': {
        'mlp_hidden_layers': [
            [128, 64],  # Previous winner
            [128, 64, 32],  # Deeper
            [256, 128],  # Wider
            [256, 128, 64]  # Wider and Deeper
        ],
        'text_projection_dim': [
            64,  # Baseline (no projection)
            32,
            256
        ],
        'dropout_rate': [0.25],  # Lock in winner
    }}

if __name__ == "__main__":
    tuner = HyperTuner.load(3)
    tuner.run(structured_grid_config, n_splits=3)
