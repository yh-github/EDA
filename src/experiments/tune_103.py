from hyper_tuner import HyperTuner
from config import EmbModel
from classifier_transformer import TransformerHyperParams

structured_grid_config = {
    'exp_params': {
        'learning_rate': [1e-4],  # Transformers prefer lower/stable LR
        'batch_size': [128],
        'epochs': [20]  # Give deeper models time to converge
    },
    'emb_params': {
        'model_name': [EmbModel.MPNET]
    },
    'feat_proc_params': {
        # Lock in the best feature set from previous runs
        'use_cyclical_dates': [True],
        'use_categorical_dates': [True],

        # A/B Test: Does the transformer need explicit amount tokens?
        # Data suggested k_top=30 was strong, despite top 1 model not using it.
        'use_categorical_amount': [False, True],
        'use_continuous_amount': [True],
        'k_top': [30],
        'n_bins': [20]
    },
    'model_params': {
        # --- The Winners ---
        'd_model': [128],
        'n_head': [8],
        'norm_first': [True],

        # --- The Tuning Targets ---
        # 1. Depth: Can we go deeper now that NormFirst stabilized training?
        'num_encoder_layers': [4, 6],

        # 2. Architecture: Deep Sets (Mean) vs Standard BERT-like (CLS)
        'pooling_strategy': ['cls', 'mean'],

        # 3. Regularization: Transformers on tabular data often need high dropout
        'dropout_rate': [0.2, 0.3]
    }
}

if __name__ == "__main__":
    tuner = HyperTuner.load(
        103,
        model_config_class=TransformerHyperParams,
        unique_cache=True,
        filter_direction=1  # Tune on the harder "Incoming" or "Outgoing" side
    )
    tuner.run(structured_grid_config, n_splits=4)