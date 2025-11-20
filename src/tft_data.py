from typing import Any

import pandas as pd
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