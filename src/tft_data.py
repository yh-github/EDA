from typing import Any
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder, EncoderNormalizer
from sklearn.decomposition import PCA
from config import FieldConfig


def prepare_tft_data(df: pd.DataFrame, field_config: FieldConfig,
                     pca_model=None, embeddings=None) -> tuple[pd.DataFrame, Any]:
    df = df.copy()

    # 1. Sort and Label
    df = df.sort_values([field_config.accountId, field_config.date]).reset_index(drop=True)
    df[field_config.label] = df[field_config.label].astype(int)

    # 2. Create Sequential Index (Sequence Order)
    df["time_idx"] = df.groupby(field_config.accountId).cumcount()

    # 3. Create PHYSICAL Time Features (The Fix)
    # We need the model to see the actual passage of time (days)
    dates = pd.to_datetime(df[field_config.date])

    # Feature A: continuous days for calculating gaps
    # We normalize per account to start at 0 to keep numbers small/stable
    df["days_from_start"] = df.groupby(field_config.accountId)[field_config.date].transform(
        lambda x: (pd.to_datetime(x) - pd.to_datetime(x).min()).dt.total_seconds() / (24 * 3600)
    )

    # Feature B: Cyclical features (Month-Day is crucial for bills)
    df["day_of_month"] = dates.dt.day.astype(str).astype("category")  # Treat as categorical

    # 4. Compress Text Embeddings
    if embeddings is not None:
        if pca_model is None:
            pca_model = PCA(n_components=16)
            compressed = pca_model.fit_transform(embeddings)
        else:
            compressed = pca_model.transform(embeddings)

        for i in range(16):
            df[f"text_pca_{i}"] = compressed[:, i]

    return df, pca_model


def build_tft_dataset(train_df_prepped: pd.DataFrame, field_config: FieldConfig, max_prediction_length=1,
                      max_encoder_length=60):  # Increased from 30 to 60
    """
    Defines the TimeSeriesDataSet with explicit Time and PCA features.
    """

    # Define feature groups
    pca_cols = [f"text_pca_{i}" for i in range(16)]

    # Define Scalers explicitly for PCA columns (important for neural nets)
    # We use Standard scaling for PCA components
    scalers = {col: EncoderNormalizer(method="standard") for col in pca_cols}

    # Add Amount scaler
    scalers[field_config.amount] = EncoderNormalizer(method="standard")
    scalers["days_from_start"] = EncoderNormalizer(method="standard")

    training = TimeSeriesDataSet(
        train_df_prepped,
        time_idx="time_idx",
        target=field_config.label,
        group_ids=[field_config.accountId],

        # --- WINDOWING (Increased) ---
        # 30 transactions might only be 3 days of coffee.
        # 60-100 gives a better chance to see the "last month" bill.
        min_encoder_length=10,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,

        # --- FEATURES ---
        static_categoricals=[field_config.accountId],

        # Known Reals: We know the date of the transaction we are predicting!
        time_varying_known_reals=["days_from_start"],
        time_varying_known_categoricals=["day_of_month"],

        # Unknown Reals: We don't know the Amount/Text of the FUTURE transaction
        # (If we are predicting "Is *this* transaction recurring?", we DO know its features.
        #  However, standard TFT forecasts t+1. Assuming you want to classify the current step
        #  based on history, these technically belong in known_reals, but unknown is safer for leakage).
        time_varying_unknown_reals=[field_config.amount] + pca_cols,

        # --- SCALERS & ENCODERS ---
        categorical_encoders={
            field_config.accountId: NaNLabelEncoder(add_nan=True),
            "day_of_month": NaNLabelEncoder(add_nan=True)
        },

        scalers=scalers,

        target_normalizer=NaNLabelEncoder(add_nan=False),

        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    return training