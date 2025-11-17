from classifier import HybridModel
from feature_processor import FeatProcParams
from hyper_tuner import HyperTuner
from config import EmbModel

# TODO grid search around best from tune1

structured_grid_config = {
    'exp_params': {
        'learning_rate': [1e-3, 5e-4],
        'batch_size': [128, 256]
    },
    'emb_params': {
        'model_name': [EmbModel.ALBERT, EmbModel.MiniLM_L12, EmbModel.FINBERT],
    },
    'feat_proc_params': {
        'use_cyclical_dates': [True, False],
        'use_categorical_amount': [False],
        'use_continuous_amount': [True, False],
        'use_categorical_dates': [True, False]

    },
    'model_params': {
        'mlp_hidden_layers': [[64], [128, 64]],
        'dropout_rate': [0.25, 0.4]
    }
}

structured_grid_config = {
    'exp_params': {
        'learning_rate': [0.001],  # Lock in winner
        'batch_size': [128],  # Lock in winner
    },
    'emb_params': {
        'model_name': [EmbModel.MiniLM_L12],  # Lock in winner
    },
    'feat_proc_params': {
        'use_cyclical_dates': [False, True],
        'use_categorical_dates': [False, True],
        'use_continuous_amount': [False, True],
        'use_categorical_amount': [False]
        # 'k_top': [10, 20, 30],
        # 'n_bins': [10, 20, 30],
    },
    'model_params': {
        'mlp_hidden_layers': [
            [128, 64],  # Previous winner
            [128, 64, 32],  # Deeper
            [256, 128],  # Wider
            [256, 128, 64]  # Wider and Deeper
        ],
        'text_projection_dim': [
            None,  # Baseline (no projection)
            128  # Project 768-dim text embedding to 128
        ],
        'dropout_rate': [0.25],  # Lock in winner
    }}

if __name__ == "__main__":
    tuner = HyperTuner.load(2)
    tuner.run(structured_grid_config, n_splits=3)
