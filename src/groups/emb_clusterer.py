import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

# --- GPU Acceleration Imports ---
try:
    import cupy as cp
    from cuml.manifold import UMAP as CuUMAP
    from cuml.cluster import HDBSCAN as CuHDBSCAN

    HAS_GPU_CLUSTERING = True
except ImportError:
    HAS_GPU_CLUSTERING = False

# --- CPU Fallback Imports ---
try:
    import umap
    import hdbscan
except ImportError:
    import umap.umap_ as umap
    import hdbscan

from common.config import FieldConfig
from common.feature_processor import HybridFeatureProcessor, FeatProcParams
from common.embedder import EmbeddingService
from tft.group import InteroperableGroup

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    groups: list[InteroperableGroup]
    clustered_df: pd.DataFrame
    metrics: dict[str, float]


class EmbClusterer:
    """
    Clusters transactions using Balanced Hybrid Embeddings (PCA-Text + Dense Features).
    Enforces strict Period Stability to distinguish 'Recurring' from 'Frequent'.
    """

    def __init__(
            self,
            field_config: FieldConfig,
            feat_params: FeatProcParams,
            emb_service: EmbeddingService,
            min_samples: int = 2,  # Lowered to capture pairs
            umap_components: int = 5,
            umap_neighbors: int = 15,
            cluster_epsilon: float = 0.0,  # Tighter clusters
            use_gpu: bool = True,
            # --- Filters ---
            max_amount_cv: float = 0.1,  # Amount Stability
            min_cycle_days: float = 6.0,  # Minimum Period
            max_period_std: float = 3.5,  # New: Period STABILITY (Standard Deviation of gaps)
            text_pca_components: int = 16  # New: Compress text to balance features
    ):
        self.field_config = field_config
        self.feat_params = feat_params
        self.emb_service = emb_service
        self.min_samples = min_samples
        self.umap_components = umap_components
        self.umap_neighbors = umap_neighbors
        self.cluster_epsilon = cluster_epsilon

        self.max_amount_cv = max_amount_cv
        self.min_cycle_days = min_cycle_days
        self.max_period_std = max_period_std
        self.text_pca_components = text_pca_components

        self.use_gpu = use_gpu and HAS_GPU_CLUSTERING
        if use_gpu and not HAS_GPU_CLUSTERING:
            logger.warning("GPU clustering requested but cuML not found. Falling back to CPU.")

    def cluster_account(self, df: pd.DataFrame) -> ClusteringResult:
        if len(df) < self.min_samples:
            return self._empty_result(df)

        # 1. Dense Features (Date/Amount)
        processor = HybridFeatureProcessor(self.feat_params, self.field_config)
        meta = processor.fit(df)
        features_df = processor.transform(df)

        dense_cols = meta.continuous_scalable_cols + meta.cyclical_cols
        X_dense = features_df[dense_cols].values

        # 2. Text Embeddings with PCA Compression
        texts = df[self.field_config.text].tolist()
        X_text_raw = self.emb_service.embed(texts)

        # We perform PCA on text to prevent it from overwhelming the dense features
        # If dataset is too small for PCA, use raw (sliced)
        n_samples = X_text_raw.shape[0]
        n_comps = min(self.text_pca_components, n_samples)

        if n_comps < 2:
            X_text_compressed = X_text_raw[:, :self.text_pca_components]
        else:
            pca = PCA(n_components=n_comps)
            X_text_compressed = pca.fit_transform(X_text_raw)

        # 3. Balanced Hybrid Vector
        # We scale dense features to have unit variance, similar to PCA output
        if X_dense.shape[1] > 0:
            scaler = StandardScaler()
            X_dense_scaled = scaler.fit_transform(X_dense)

            # Optional: Weighting can be applied here.
            # Currently treating 16 Text Dims roughly equal to 10 Dense Dims.
            X_hybrid = np.hstack([X_text_compressed, X_dense_scaled])
        else:
            X_hybrid = X_text_compressed

        # 4. UMAP
        n_neighbors = min(self.umap_neighbors, len(df) - 1)
        if n_neighbors < 2:
            return self._empty_result(df)

        if self.use_gpu:
            reducer = CuUMAP(
                n_neighbors=n_neighbors,
                n_components=self.umap_components,
                min_dist=0.0,
                metric='cosine',  # Cosine on the hybrid vector handles the mix well
                random_state=42
            )
        else:
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=self.umap_components,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )

        embedding_projection = reducer.fit_transform(X_hybrid)

        # 5. HDBSCAN
        if self.use_gpu:
            clusterer = CuHDBSCAN(
                min_cluster_size=self.min_samples,
                metric='euclidean',
                cluster_selection_epsilon=self.cluster_epsilon,
                gen_min_span_tree=True
            )
            labels = clusterer.fit_predict(embedding_projection)
            probs = getattr(clusterer, 'probabilities_', np.ones_like(labels))
            if hasattr(labels, 'get'): labels = labels.get()
            if hasattr(probs, 'get'): probs = probs.get()
        else:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_samples,
                metric='euclidean',
                cluster_selection_epsilon=self.cluster_epsilon,
                prediction_data=True
            )
            labels = clusterer.fit_predict(embedding_projection)
            probs = clusterer.probabilities_

        # 6. Extract & Validate Groups
        df_out = df.copy()
        df_out['cluster_label'] = labels
        df_out['cluster_prob'] = probs
        df_out['prediction'] = 0

        groups = self._extract_groups(df_out, labels, probs)

        for grp in groups:
            if grp.is_recurring:
                cluster_id = int(grp.group_id.split("_")[1])
                df_out.loc[df_out['cluster_label'] == cluster_id, 'prediction'] = 1

        metrics = self._evaluate(df_out)
        return ClusteringResult(groups=groups, clustered_df=df_out, metrics=metrics)

    def _extract_groups(self, df: pd.DataFrame, labels: np.ndarray, probs: np.ndarray) -> list[InteroperableGroup]:
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        results = []
        for label in unique_labels:
            mask = (labels == label)
            cluster_rows = df[mask]

            # Metadata
            text_mode = cluster_rows[self.field_config.text].mode()
            description = text_mode[0] if not text_mode.empty else "Unknown"

            amounts = cluster_rows[self.field_config.amount]
            median_amount = float(amounts.median())

            # Amount Stability
            amt_mean_abs = amounts.abs().mean()
            amount_cv = amounts.std() / amt_mean_abs if amt_mean_abs > 1e-3 else 0.0

            # Cycle Logic (Robust)
            dates = pd.to_datetime(cluster_rows[self.field_config.date]).sort_values()
            cycle_days = 0.0
            period_std = 999.0  # High default variance

            if len(dates) > 1:
                diffs = dates.diff().dt.days.dropna()
                cycle_days = float(diffs.median())
                if len(diffs) > 1:
                    period_std = float(diffs.std())
                else:
                    # If only 2 transactions, we can't measure stability of the interval
                    # We assume it's stable if the amount is stable
                    period_std = 0.0

            confidence = float(probs[mask].mean())

            # --- Filter Logic ---
            is_valid = True

            # 1. Variance Check (Ad-hoc spending)
            if amount_cv > self.max_amount_cv:
                is_valid = False

            # 2. Min Cycle (Bursts)
            if cycle_days < self.min_cycle_days:
                is_valid = False

            # 3. Period Stability (Regularity)
            # A true subscription has consistent intervals.
            if period_std > self.max_period_std:
                is_valid = False

            group = InteroperableGroup(
                group_id=f"cluster_{label}",
                description=description,
                amount_stats=median_amount,
                cycle_days=cycle_days,
                confidence=confidence,
                is_recurring=is_valid
            )
            results.append(group)

        return results

    def _evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        y_true = df[self.field_config.label].astype(int).values
        y_pred = df['prediction'].values
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        return {"f1": float(f1), "precision": float(p), "recall": float(r)}

    def _empty_result(self, df: pd.DataFrame) -> ClusteringResult:
        df_out = df.copy()
        df_out['cluster_label'] = -1
        df_out['cluster_prob'] = 0.0
        df_out['prediction'] = 0
        metrics = self._evaluate(df_out)
        return ClusteringResult(groups=[], clustered_df=df_out, metrics=metrics)