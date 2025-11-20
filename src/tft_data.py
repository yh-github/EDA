from typing import Any
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder, EncoderNormalizer
from sklearn.decomposition import PCA

from config import FieldConfig


def prepare_tft_data(df: pd.DataFrame, field_config: FieldConfig,
                     pca_model=None, embeddings=None) -> tuple[pd.DataFrame, Any]:
    """
    Now accepts raw embeddings and compresses them for the TFT.
    """
    df = df.copy()

    # ... (Sort and Cast Target - Existing Code) ...
    df = df.sort_values([field_config.accountId, field_config.date]).reset_index(drop=True)
    df[field_config.label] = df[field_config.label].astype(int)
    df["time_idx"] = df.groupby(field_config.accountId).cumcount()

    # --- NEW: Compress Text Embeddings ---
    if embeddings is not None:
        # If training, fit PCA. If validation/test, use existing PCA.
        if pca_model is None:
            pca_model = PCA(n_components=16)  # Compress 768 -> 16
            compressed = pca_model.fit_transform(embeddings)
        else:
            compressed = pca_model.transform(embeddings)

        # Add these 16 columns to the DataFrame
        for i in range(16):
            df[f"text_pca_{i}"] = compressed[:, i]

    return df, pca_model


def build_tft_dataset(train_df_prepped: pd.DataFrame, field_config: FieldConfig, max_prediction_length=1,
                      max_encoder_length=30):
    """
    Defines the TimeSeriesDataSet using the prepared training data.
    """

    # --- FIX: Define the feature columns explicitly ---
    # These match the columns created in prepare_tft_data
    pca_cols = [f"text_pca_{i}" for i in range(16)]

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

        # --- FIX: Add pca_cols to the inputs ---
        # Previously, this was just [field_config.amount]
        time_varying_unknown_reals=[field_config.amount] + pca_cols,

        # --- LAGS ---
        lags={
            field_config.amount: [1, 2, 3, 4, 5, 10]
        },

        # --- SCALERS & ENCODERS ---
        categorical_encoders={
            field_config.accountId: NaNLabelEncoder(add_nan=True)
        },

        scalers={
            field_config.amount: EncoderNormalizer(method="standard"),
            # Note: PCA columns are already somewhat normalized,
            # but passing them to EncoderNormalizer is safe and standard practice for TFT.
        },

        target_normalizer=NaNLabelEncoder(add_nan=False),

        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    return training