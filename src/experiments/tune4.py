from glob.hyper_tuner import HyperTuner
from common.config import EmbModel

structured_grid_config = {
    'exp_params': {
        'learning_rate': [1e-3, 5e-4],
        'batch_size': [64, 128],
    },
    'emb_params': {
        'model_name': [EmbModel.MPNET, EmbModel.MiniLM_L12],
    },
    'feat_proc_params': {
        'use_cyclical_dates': [False, True],
        'use_categorical_dates': [False, True],
        'use_continuous_amount': [False, True],
        'use_categorical_amount': [False],
        'k_top': [0],
        'n_bins': [0]
    },
    'model_params': {
        'mlp_hidden_layers': [[128, 64, 32]],
        'dropout_rate': [0.25, 0.4],
        'text_projection_dim': [
            None,
            128
        ]
    }
}

if __name__ == "__main__":
    tuner = HyperTuner.load(4, unique_cache=True)
    tuner.run(structured_grid_config, n_splits=4)

