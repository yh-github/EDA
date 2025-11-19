import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder, EncoderNormalizer
from config import FieldConfig


def prepare_tft_data(df: pd.DataFrame, field_config: FieldConfig) -> pd.DataFrame:
    """
    Prepares a raw DataFrame for TFT by adding time_idx and casting types.
    Must be called on Train, Val, and Test sets independently.
    """
    df = df.copy()

    # 1. Sort by Account and Date (Critical for time_idx)
    df = df.sort_values([field_config.accountId, field_config.date]).reset_index(drop=True)

    # 2. Cast Target to Integer (for Classification)
    df[field_config.label] = df[field_config.label].astype(int)

    # 3. Create Time Index (0, 1, 2... per account)
    df["time_idx"] = df.groupby(field_config.accountId).cumcount()

    return df


def build_tft_dataset(train_df_prepped: pd.DataFrame, field_config: FieldConfig, max_prediction_length=1,
                      max_encoder_length=30):
    """
    Defines the TimeSeriesDataSet using the prepared training data.
    """
    training = TimeSeriesDataSet(
        train_df_prepped,
        time_idx="time_idx",
        target=field_config.label,
        group_ids=[field_config.accountId],

        # --- WINDOWING ---
        min_encoder_length=3,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,

        # --- FEATURES ---
        static_categoricals=[field_config.accountId],
        time_varying_unknown_reals=[field_config.amount],

        # --- LAGS ---
        lags={
            field_config.amount: [1, 2, 3, 4, 5, 10]
        },

        # --- SCALERS & ENCODERS (The Fixes) ---

        # 1. Fix for 'Unknown category' error:
        # Allow the encoder to handle unseen Account IDs (in Val/Test) by mapping them to a special 'NaN' token.
        categorical_encoders={
            field_config.accountId: NaNLabelEncoder(add_nan=True)
        },

        # 2. Fix for sklearn feature name error:
        scalers={
            field_config.amount: EncoderNormalizer(method="standard")
        },

        # 3. Fix for Classification Target:
        target_normalizer=NaNLabelEncoder(add_nan=False),

        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    return training