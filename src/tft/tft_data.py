import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder, EncoderNormalizer
from sklearn.decomposition import PCA
from common.config import FieldConfig
from common.embedder import EmbeddingService
from common.feature_processor import HybridFeatureProcessor, FeatProcParams, FeatureMetadata


def prepare_tft_data(
        df: pd.DataFrame,
        field_config: FieldConfig,
        feat_params: FeatProcParams = None,
        embedding_service: EmbeddingService = None,
        pca_model: PCA = None,
        processor: HybridFeatureProcessor = None,
        fit_processor: bool = False,
        filter_direction: int = 1
) -> tuple[pd.DataFrame, PCA, HybridFeatureProcessor, FeatureMetadata]:
    """
    Prepares data for TFT by:
    1. Filtering for Outgoing (Positive) or Incoming (Negative) transactions.
    2. Sorting by Account/Date (CRITICAL to do this before embedding alignment).
    3. Generating/Compressing Text Embeddings.
    4. Running HybridFeatureProcessor for Date/Amount features.
    """
    pca_components = 16

    df = df.copy()

    # 1. Filter based on direction
    # Using explicit copy to avoid SettingWithCopy warnings
    if filter_direction > 0:
        df = df[df[field_config.amount] > 0].copy()
    elif filter_direction < 0:
        df = df[df[field_config.amount] < 0].copy()

    # 2. Sort and Label (CRITICAL: Must be done before aligning embeddings)
    df = df.sort_values([field_config.accountId, field_config.date]).reset_index(drop=True)
    df[field_config.label] = df[field_config.label].astype(int)

    # 3. Create Time Index
    df["time_idx"] = df.groupby(field_config.accountId).cumcount()

    # 4. Hybrid Feature Processor (Date/Amount Features)
    if fit_processor:
        if feat_params is None:
            raise ValueError("feat_params required when fit_processor=True")
        processor = HybridFeatureProcessor.create(feat_params, field_config)
        meta = processor.fit(df)
    else:
        if processor is None:
            raise ValueError("processor required when fit_processor=False")
        # We need metadata to know which columns are what
        meta = processor._build_meta()

    features_df = processor.transform(df)

    # Merge features back into main DF
    # We use index join as both are reset
    df = pd.concat([df, features_df], axis=1)

    # --- FIX: Cast Categoricals to String ---
    # PyTorch Forecasting requires categoricals to be strings to learn embeddings properly
    if meta is not None:
        for cat_col in meta.categorical_features.keys():
            if cat_col in df.columns:
                df[cat_col] = df[cat_col].astype(str)

    # 5. Text Embeddings & PCA
    # We generate embeddings AFTER sorting to ensure alignment
    if embedding_service is not None:
        texts = df[field_config.text].tolist()
        # The service handles caching, so this is fast if repeated
        embeddings = embedding_service.embed(texts)

        if pca_model is None:
            pca_model = PCA(n_components=pca_components)
            compressed = pca_model.fit_transform(embeddings)
        else:
            compressed = pca_model.transform(embeddings)

        for i in range(16):
            df[f"text_pca_{i}"] = compressed[:, i]

    return df, pca_model, processor, meta


def build_tft_dataset(
       train_df_prepped: pd.DataFrame,
       field_config: FieldConfig,
       feature_metadata: FeatureMetadata,
       max_encoder_length:int,
       max_prediction_length=1,
):
    """
    Defines the TimeSeriesDataSet using metadata from HybridFeatureProcessor.
    """

    # --- 1. Define Real (Continuous) Features ---
    # PCA Columns
    pca_cols = [c for c in train_df_prepped.columns if "text_pca_" in c]

    # Processor Continuous Cols (e.g. log_abs_amount)
    # We treat them as KNOWN reals because for classification at time t,
    # we know the amount at time t.
    processor_reals = feature_metadata.continuous_scalable_cols

    # Processor Cyclical Cols (sin/cos dates)
    processor_cyclical = feature_metadata.cyclical_cols

    # Combine all known reals
    known_reals = pca_cols + processor_reals + processor_cyclical

    # --- 2. Define Categorical Features ---
    # Processor Categoricals (e.g. amount_token_id, is_positive)
    processor_cats = list(feature_metadata.categorical_features.keys())

    known_categoricals = processor_cats

    # --- 3. Define Scalers ---
    # TFT needs to know how to normalize these.
    scalers = {}

    # Standard scale PCA and Continuous Processor features
    for col in (pca_cols + processor_reals):
        scalers[col] = EncoderNormalizer(method="standard")

    # Cyclical features are already -1 to 1, but "standard" or "identity" is fine.
    # Leaving them to default (GroupNormalizer) or forcing Identity often works best for sin/cos.
    # Here we stick to None (default) or Standard.
    for col in processor_cyclical:
        scalers[col] = EncoderNormalizer(method="standard")

    # --- 4. Build Dataset ---
    training = TimeSeriesDataSet(
        train_df_prepped,
        time_idx="time_idx",
        target=field_config.label,
        group_ids=[field_config.accountId],

        min_encoder_length=3,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,

        # static_categoricals=[field_config.accountId],

        # Everything is "Known" at the time of transaction for classification purposes
        time_varying_known_reals=known_reals,
        time_varying_known_categoricals=known_categoricals,

        # We usually don't have "unknown" reals for classification tasks
        # unless we are strictly forecasting t+1 without knowing t+1 features.
        time_varying_unknown_reals=[],
        time_varying_unknown_categoricals=[],

        categorical_encoders={
            field_config.accountId: NaNLabelEncoder(add_nan=True),
            # Add encoders for processor categoricals (amount tokens, etc)
            **{k: NaNLabelEncoder(add_nan=True) for k in known_categoricals}
        },

        scalers=scalers,
        target_normalizer=NaNLabelEncoder(add_nan=False),  # Binary classification

        add_relative_time_idx=True,
        add_target_scales=False,  # Not needed for classification usually
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    return training