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

if __name__ == "__main__":
    tuner = HyperTuner.load(2)
    tuner.run(structured_grid_config, n_splits=3)
