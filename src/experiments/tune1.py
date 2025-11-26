from pointwise.hyper_tuner import HyperTuner
from common.config import EmbModel

structured_grid_config = {
    'exp_params': {
        'learning_rate': [1e-3, 5e-4],
        'batch_size': [128, 256],
    },
    'emb_params': {
        'model_name': [EmbModel.ALBERT, EmbModel.MiniLM_L12, EmbModel.FINBERT],
    },
    'feat_proc_params': {
        'k_top': [20, 50, 100],
        'n_bins': [20, 50, 100]
    },
    'model_params': {
        'mlp_hidden_layers': [[64], [128, 64]],
        'dropout_rate': [0.25, 0.4]
    }
}

if __name__ == "__main__":
    tuner = HyperTuner.load(1)
    tuner.run(structured_grid_config, n_splits=3)

