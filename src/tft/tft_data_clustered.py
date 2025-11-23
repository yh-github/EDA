import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder, EncoderNormalizer
from sklearn.decomposition import PCA

from common.config import FieldConfig
from common.embedder import EmbeddingService
from common.feature_processor import HybridFeatureProcessor, FeatProcParams, FeatureMetadata


def assign_amount_clusters_with_overlap(
        df: pd.DataFrame,
        field_config: FieldConfig,
        tol_abs: float = 1.0,
        tol_pct: float = 0.05,
        min_bin_size: int = 2
) -> pd.DataFrame:
    """
    Generates candidate clusters based purely on Amount Similarity.
    Allows OVERLAP: A single transaction can appear in multiple amount bins.
    """
    new_rows = []

    # Process each account independently
    for acc_id, group in df.groupby(field_config.accountId):
        # Work with numpy arrays for speed
        amounts = group[field_config.amount].values
        ids = group[field_config.trId].values

        # 1. Identify Seeds (Unique amounts are potential centers)
        # We round to 2 decimals to avoid float noise
        seeds = np.unique(np.round(amounts, 2))

        created_clusters = []  # To check for subsets/duplicates

        for seed in seeds:
            # 2. Find all transactions close to this seed
            # Logic: abs_diff <= tol_abs OR pct_diff <= tol_pct
            abs_diff = np.abs(amounts - seed)
            mask = (abs_diff <= tol_abs) | (abs_diff <= (np.abs(seed) * tol_pct))

            # Indices of matching transactions
            match_indices = np.where(mask)[0]

            if len(match_indices) < min_bin_size:
                continue

            # 3. Deduplication Strategy
            # If this new cluster is a SUBSET of an existing cluster, ignore it.
            # (e.g. seed=10.0 found [10.0, 10.1]. seed=10.1 found [10.0, 10.1, 10.2].
            # We keep the larger one).
            # Note: This is a heuristic. For perfect "all possible views", we keep everything,
            # but that explodes data size. Subset pruning is a safe middle ground.

            current_id_set = set(ids[match_indices])
            is_subset = False
            for existing_set in created_clusters:
                if current_id_set.issubset(existing_set):
                    is_subset = True
                    break

            if is_subset:
                continue

            # Add to our list of valid clusters
            created_clusters.append(current_id_set)

            # 4. Create Rows
            # We assign a deterministic ID based on the seed amount
            cluster_id = f"{seed:.2f}"
            unique_grp_id = f"{acc_id}_amt_{cluster_id}"

            # Extract the actual rows
            cluster_rows = group.iloc[match_indices].copy()
            cluster_rows['global_group_id'] = unique_grp_id
            cluster_rows['cluster_seed_amount'] = seed  # Metadata

            new_rows.append(cluster_rows)

    if not new_rows:
        raise ValueError("No amount clusters found! Check data or tolerances.")

    # Combine all expanded clusters
    # Note: The index might be duplicated because we duplicated transactions.
    # We reset it.
    expanded_df = pd.concat(new_rows).reset_index(drop=True)

    return expanded_df


def prepare_clustered_tft_data(
        df: pd.DataFrame,
        field_config: FieldConfig,
        feat_params: FeatProcParams = None,
        embedding_service: EmbeddingService = None,
        pca_model: PCA = None,
        processor: HybridFeatureProcessor = None,
        fit_processor: bool = False
) -> tuple[pd.DataFrame, PCA, HybridFeatureProcessor, FeatureMetadata]:
    df = df.copy()

    # 1. Run Amount Clustering (Physical Binning)
    print("Running Amount Binning (with Overlap)...")
    # We generate a new, larger DataFrame where rows are duplicated across bins
    df = assign_amount_clusters_with_overlap(df, field_config)

    print(f"Expanded Dataset: {len(df)} rows (includes overlaps)")

    # 3. Sort and Label
    #    Sort by GROUP (Bin), then Date
    df = df.sort_values(['global_group_id', field_config.date]).reset_index(drop=True)
    df[field_config.label] = df[field_config.label].astype(int)

    # 4. Create Time Index per GROUP
    df["time_idx"] = df.groupby("global_group_id").cumcount()

    # 5. Feature Processing
    if fit_processor:
        processor = HybridFeatureProcessor.create(feat_params, field_config)
        meta = processor.fit(df)
    else:
        meta = processor._build_meta()

    features_df = processor.transform(df)
    df = pd.concat([df, features_df], axis=1)

    # 6. Cast Categoricals
    if meta is not None:
        for cat_col in meta.categorical_features.keys():
            if cat_col in df.columns:
                df[cat_col] = df[cat_col].astype(str)

    # 7. Text Embeddings & PCA
    if embedding_service is not None:
        texts = df[field_config.text].tolist()
        embeddings = embedding_service.embed(texts)

        if pca_model is None:
            pca_model = PCA(n_components=16)
            compressed = pca_model.fit_transform(embeddings)
        else:
            compressed = pca_model.transform(embeddings)

        for i in range(16):
            df[f"text_pca_{i}"] = compressed[:, i]

    return df, pca_model, processor, meta


def build_clustered_tft_dataset(
        train_df_prepped: pd.DataFrame,
        field_config: FieldConfig,
        feature_metadata: FeatureMetadata,
        max_encoder_length: int = 64
):
    """
    Builds dataset using 'global_group_id' (Amount Bin) as the series identifier.
    """

    pca_cols = [c for c in train_df_prepped.columns if "text_pca_" in c]
    processor_reals = feature_metadata.continuous_scalable_cols
    processor_cyclical = feature_metadata.cyclical_cols

    # We add 'cluster_seed_amount' as a feature so the model knows the "target" amount
    # But we need to scale it or log it. For now, let's rely on the 'amount' column itself.

    known_reals = pca_cols + processor_reals + processor_cyclical
    processor_cats = list(feature_metadata.categorical_features.keys())
    known_categoricals = processor_cats

    scalers = {}
    for col in (pca_cols + processor_reals):
        scalers[col] = EncoderNormalizer(method="standard")
    for col in processor_cyclical:
        scalers[col] = EncoderNormalizer(method="standard")

    training = TimeSeriesDataSet(
        train_df_prepped,
        time_idx="time_idx",
        target=field_config.label,

        # The Bin ID is the "Series"
        group_ids=["global_group_id"],

        min_encoder_length=3,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=1,

        static_categoricals=[field_config.accountId],

        time_varying_known_reals=known_reals,
        time_varying_known_categoricals=known_categoricals,

        categorical_encoders={
            field_config.accountId: NaNLabelEncoder(add_nan=True),
            "global_group_id": NaNLabelEncoder(add_nan=True),
            **{k: NaNLabelEncoder(add_nan=True) for k in known_categoricals}
        },

        scalers=scalers,
        target_normalizer=NaNLabelEncoder(add_nan=False),
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    return training