from hyper_tuner import HyperTuner
from config import EmbModel

structured_grid_config = {
    'exp_params': {
        'learning_rate': [1e-3, 5e-4],
        'batch_size': [128]
    },
    'emb_params': {
        'model_name': [EmbModel.MPNET]
    },
    'feat_proc_params': {
        'use_cyclical_dates': [True, False],
        'use_categorical_dates': [True, False],
        'use_continuous_amount': [True, False],
        'use_is_positive': [False],
        'use_categorical_amount': [False],
        # 'k_top': [30, 50],
        # 'n_bins': [30, 50]
    },
    'model_params': {
        'mlp_hidden_layers': [
            [256, 128, 64],
            [512, 256, 128]
        ],
        'dropout_rate': [0.25, 0.4],
        'text_projection_dim': [None, 128]
    }
}

if __name__ == "__main__":
    tuner = HyperTuner.load(10, unique_cache=True, filter_direction=1)
    tuner.run(structured_grid_config, n_splits=4)
