import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from common.config import FieldConfig
from common.embedder import EmbeddingService
from common.feature_processor import FeatProcParams, HybridFeatureProcessor
from tft.tft_data_clustered import build_clustered_tft_dataset
from tft.tft_experiment import TFTTuningExperiment

# --- CONFIG ---
MAX_ENCODER_LEN = 64
BATCH_SIZE = 2048
MAX_EPOCHS = 20
N_TRIALS = 20
STUDY_NAME = "tft_non_overlapping_v1.1"
BEST_MODEL_PATH = "cache/tft_models/best_tune5_model.pt"

logger = logging.getLogger(__name__)


def assign_non_overlapping_amount_clusters(
        df: pd.DataFrame,
        field_config: FieldConfig,
        gap_threshold_abs: float = 1.0,
        gap_threshold_pct: float = 0.05,
        min_cluster_size: int = 2
) -> pd.DataFrame:
    """
    Groups transactions by amount proximity into NON-OVERLAPPING clusters.

    Algorithm:
    1. Group by Account.
    2. Sort transactions by Amount.
    3. Calculate difference between consecutive amounts.
    4. Break cluster if diff > threshold.
    5. Filter out small clusters (noise).
    """
    clustered_rows = []

    # Process each account independently
    for acc_id, group in df.groupby(field_config.accountId):
        # Sort by amount to find 1D density
        sorted_group = group.sort_values(field_config.amount)
        amounts = sorted_group[field_config.amount].values
        ids = sorted_group[field_config.trId].values

        if len(amounts) < min_cluster_size:
            continue

        # Calculate gaps: diff[i] = amount[i+1] - amount[i]
        diffs = np.diff(amounts)
        abs_diffs = np.abs(diffs)

        # Determine thresholds relative to the magnitude of the amounts
        magnitudes = np.abs(amounts[:-1])

        # A gap exists if it exceeds BOTH the absolute AND percentage tolerance
        is_gap = (abs_diffs > gap_threshold_abs) & \
                 (abs_diffs > (magnitudes * gap_threshold_pct))

        # Identify start indices of new clusters (shift by 1 because diff is N-1)
        split_indices = np.where(is_gap)[0] + 1

        # Split the array into clusters
        cluster_slices = np.split(np.arange(len(amounts)), split_indices)

        for idx_list in cluster_slices:
            if len(idx_list) < min_cluster_size:
                continue

            cluster_ids = ids[idx_list]
            cluster_amounts = amounts[idx_list]

            # Deterministic Group ID based on Median Amount
            median_amt = np.median(cluster_amounts)
            unique_grp_id = f"{acc_id}_amt_{median_amt:.2f}"

            # Extract original rows
            subset = group[group[field_config.trId].isin(cluster_ids)].copy()
            subset['global_group_id'] = unique_grp_id
            subset['cluster_seed_amount'] = median_amt

            clustered_rows.append(subset)

    if not clustered_rows:
        logger.warning("No clusters found with current settings!")
        return pd.DataFrame(columns=df.columns)

    return pd.concat(clustered_rows).reset_index(drop=True)


def prepare_non_overlapping_data(
        df: pd.DataFrame,
        field_config: FieldConfig,
        feat_params: FeatProcParams = None,
        embedding_service: EmbeddingService = None,
        pca_model: PCA = None,
        processor: HybridFeatureProcessor = None,
        fit_processor: bool = False
):
    """
    Custom preparation pipeline for Tune 4.
    Replaces the 'Overlap' binning with 'Non-Overlap' clustering.
    """
    df = df.copy()

    # 1. Non-Overlapping Clustering
    logger.info("Running Non-Overlapping Amount Clustering...")
    df = assign_non_overlapping_amount_clusters(
        df, field_config,
        min_cluster_size=3
    )
    logger.info(f"Clustered Data Size: {len(df)} rows (subset of original)")

    # 2. Sort and Label (Sort by Group, then Date)
    df = df.sort_values(['global_group_id', field_config.date]).reset_index(drop=True)
    df[field_config.label] = df[field_config.label].astype(int)

    # 3. Create Time Index
    df["time_idx"] = df.groupby("global_group_id").cumcount()

    # 4. Feature Processing (Standard)
    if fit_processor:
        processor = HybridFeatureProcessor.create(feat_params, field_config)
        meta = processor.fit(df)
    else:
        meta = processor._build_meta()

    features_df = processor.transform(df)
    df = pd.concat([df, features_df], axis=1)

    # 5. Categoricals to String
    if meta is not None:
        for cat_col in meta.categorical_features.keys():
            if cat_col in df.columns:
                df[cat_col] = df[cat_col].astype(str)

    # 6. Text Embeddings & PCA
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


if __name__ == "__main__":
    # Standard Search Space
    search_space = {
        "hidden_size": lambda trial: trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "dropout": lambda trial: trial.suggest_float("dropout", 0.1, 0.4)
    }

    experiment = TFTTuningExperiment(
        study_name=STUDY_NAME,
        best_model_path=BEST_MODEL_PATH,
        max_encoder_len=MAX_ENCODER_LEN,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        n_trials=N_TRIALS,
        search_space=search_space,

        # Inject our NEW Non-Overlapping Strategy
        prepare_data_fn=prepare_non_overlapping_data,

        # Reuse the Clustered Dataset Builder (compatible with global_group_id)
        build_dataset_fn=build_clustered_tft_dataset,

        # Disabled per your request: Clusters are unique, so no aggregation needed.
        use_aggregation=False,

        feat_params=FeatProcParams(
            use_is_positive=False, use_categorical_dates=True, use_cyclical_dates=True,
            use_continuous_amount=True, use_categorical_amount=False, k_top=0, n_bins=0,
            use_behavioral_features=True
        )
    )

    experiment.run()