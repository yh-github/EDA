import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder, EncoderNormalizer
from config import FieldConfig


def build_tft_dataset(df: pd.DataFrame, field_config: FieldConfig, max_prediction_length=1, max_encoder_length=30):
    # 1. Ensure proper sorting and types
    df = df.sort_values([field_config.accountId, field_config.date]).reset_index(drop=True)

    # CRITICAL: Ensure target is an integer for classification
    df[field_config.label] = df[field_config.label].astype(int)

    # Create time index
    df["time_idx"] = df.groupby(field_config.accountId).cumcount()

    # 2. Define the dataset
    training = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=field_config.label,
        group_ids=[field_config.accountId],

        # Windowing
        min_encoder_length=10,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,

        # Variables
        static_categoricals=[field_config.accountId],
        time_varying_known_categoricals=[],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            field_config.amount,
        ],

        # Lags
        lags={
            field_config.amount: [1, 2, 3, 4, 5, 10]
        },

        # --- SCALERS (The Fix) ---
        # Explicitly use EncoderNormalizer for 'amount' to avoid sklearn name mismatch errors on lags.
        # We use standard scaling (method='standard') because amounts can be negative.
        scalers={
            field_config.amount: EncoderNormalizer(method="standard")
        },

        # Target Encoder
        target_normalizer=NaNLabelEncoder(add_nan=False),

        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    return training