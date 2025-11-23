from glob.hyper_tuner import HyperTuner
from common.config import EmbModel

structured_grid_config = {
    'exp_params': {
        'learning_rate': [1e-3],
        'batch_size': [128]
    },
    'emb_params': {
        'model_name': [EmbModel.MPNET]
    },
    'feat_proc_params': {
        'use_cyclical_dates': [True],
        'use_categorical_dates': [True],
        'use_continuous_amount': [True],
        'use_is_positive': [False],
        'use_categorical_amount': [True],
        'k_top': [10,20],
        'n_bins': [10,20]
    },
    'model_params': {
        'mlp_hidden_layers': [[128, 64, 32]],
        'dropout_rate': [0.25],
        'text_projection_dim': [None]
    }
}

if __name__ == "__main__":
    tuner7 = HyperTuner.load(7, unique_cache=True, filter_direction=1)
    tuner7.run(structured_grid_config, n_splits=4)

    tuner8 = HyperTuner.load(8, unique_cache=True, filter_direction=-1)
    tuner8.run(structured_grid_config, n_splits=4)
