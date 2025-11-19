import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from config import FieldConfig


def build_tft_dataset(df: pd.DataFrame, field_config: FieldConfig, max_prediction_length=1, max_encoder_length=30):
    # 1. Create a strictly increasing time index for each account
    # (Assumes df is already sorted by Date)
    df = df.sort_values([field_config.accountId, field_config.date]).reset_index(drop=True)
    df["time_idx"] = df.groupby(field_config.accountId).cumcount()

    # 2. Define the dataset
    training = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=field_config.label,
        group_ids=[field_config.accountId],

        # HISTORY WINDOW: Look back 30 steps
        min_encoder_length=10,
        max_encoder_length=max_encoder_length,

        # PREDICTION: Predict 1 step ahead (the current transaction)
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,

        # VARIABLE TYPES
        static_categoricals=[field_config.accountId],
        time_varying_known_categoricals=[
            # "day_of_week" etc. if you have them
        ],
        time_varying_unknown_categoricals=[
            # Your text embeddings should technically be handled differently,
            # but TFT handles categorical inputs natively.
            # For embeddings, we might pass them as "reals" if pre-computed.
        ],
        time_varying_unknown_reals=[
            field_config.amount,
            # Add your embedding dimensions here if passing as dense vectors
        ],

        # AUTOMATIC LAGS (The "Cheat Code")
        # This tells the model: "Look at the amount from 1 step ago, 2 steps ago..."
        lags={
            field_config.amount: [1, 2, 3, 4, 5, 10]
        },

        # AUTOMATIC SCALING
        target_normalizer=GroupNormalizer(
            groups=[field_config.accountId], transformation="softplus"
        ),  # Use softplus for positive targets or None for binary classification
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    return training