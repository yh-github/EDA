from hyper_tuner import HyperTuner
from config import EmbModel
from classifier_transformer import TransformerHyperParams

# 1. Define a grid using Transformer parameters
structured_grid_config = {
    'exp_params': {
        'learning_rate': [0.001, 0.0005],
        'batch_size': [128],
    },
    'emb_params': {
        'model_name': [EmbModel.MPNET],
    },
    'feat_proc_params': {
        'use_cyclical_dates': [True],
        'use_categorical_dates': [True],
        'use_continuous_amount': [True],
        'use_categorical_amount': [False]
    },
    'model_params': { # These keys are now for TransformerHyperParams
        'd_model': [128, 256],
        'n_head': [4, 8],
        'num_encoder_layers': [2, 4],
        'final_mlp_layers': [[64], [128, 64]],
        'dropout_rate': [0.25],
    }}

if __name__ == "__main__":
    tuner = HyperTuner.load(
        101,
        model_config_class=TransformerHyperParams,
        filter_direction=1
    )
    tuner.run(structured_grid_config, n_splits=4)