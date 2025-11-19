import pytorch_lightning as pl
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import CrossEntropy
from config import ExperimentConfig


def create_tft_model(training_dataset, exp_config: ExperimentConfig):
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,

        # TUNING KNOBS
        learning_rate=exp_config.learning_rate,
        hidden_size=64,  # d_model
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,

        # ARCHITECTURE
        output_size=2,  # Binary classification (Recurring vs Not)
        loss=CrossEntropy(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    return tft
