import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
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
    # Handle cases where umap is installed as umap-learn
    import umap.umap_ as umap
    import hdbscan

from common.config import FieldConfig
from common.feature_processor import HybridFeatureProcessor, FeatProcParams
from common.embedder import EmbeddingService
from tft.group import InteroperableGroup

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Holds the results of the clustering process for an account."""
    groups: list[InteroperableGroup]
    clustered_df: pd.DataFrame  # Original DF with 'cluster_label' and 'prediction'
    metrics: dict[str, float]  # F1, Precision, Recall


class EmbClusterer:
    """
    Clusters transactions for a single account using Hybrid Embeddings.
    Supports GPU acceleration via RAPIDS (cuML) for UMAP and HDBSCAN.
    """

    def __init__(
            self,
            field_config: FieldConfig,
            feat_params: FeatProcParams,
            emb_service: EmbeddingService,
            min_samples: int = 3,
            umap_components: int = 5,
            umap_neighbors: int = 15,
            cluster_epsilon: float = 0.5,
            use_gpu: bool = True
    ):
        self.field_config = field_config
        self.feat_params = feat_params
        self.emb_service = emb_service
        self.min_samples = min_samples
        self.umap_components = umap_components
        self.umap_neighbors = umap_neighbors
        self.cluster_epsilon = cluster_epsilon

        # Check GPU availability
        self.use_gpu = use_gpu and HAS_GPU_CLUSTERING
        if use_gpu and not HAS_GPU_CLUSTERING:
            logger.warning("GPU clustering requested but cuML/cupy not found. Falling back to CPU.")
        elif use_gpu:
            logger.info("EmbClusterer: GPU Acceleration Enabled (cuML).")

    def cluster_account(self, df: pd.DataFrame) -> ClusteringResult:
        """
        Main entry point: Features -> Embed -> Cluster -> Extract -> Evaluate.
        """
        if len(df) < self.min_samples:
            # logger.debug(f"Not enough samples ({len(df)}) to cluster.")
            return self._empty_result(df)

        # 1. Feature Generation (Amount & Date)
        processor = HybridFeatureProcessor(self.feat_params, self.field_config)
        meta = processor.fit(df)
        features_df = processor.transform(df)

        # Select continuous/dense features
        dense_cols = meta.continuous_scalable_cols + meta.cyclical_cols
        X_dense = features_df[dense_cols].values

        # 2. Text Embeddings
        # EmbeddingService handles its own GPU usage via PyTorch
        texts = df[self.field_config.text].tolist()
        X_text = self.emb_service.embed(texts)

        # 3. Construct Hybrid Embedding Vector
        if X_dense.shape[1] > 0:
            scaler = StandardScaler()
            X_dense_scaled = scaler.fit_transform(X_dense)
            X_hybrid = np.hstack([X_text, X_dense_scaled])
        else:
            X_hybrid = X_text

        # 4. UMAP Projection (Dimensionality Reduction)
        n_neighbors = min(self.umap_neighbors, len(df) - 1)
        if n_neighbors < 2:
            return self._empty_result(df)

        if self.use_gpu:
            # --- GPU UMAP ---
            reducer = CuUMAP(
                n_neighbors=n_neighbors,
                n_components=self.umap_components,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )
        else:
            # --- CPU UMAP ---
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=self.umap_components,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )

        embedding_projection = reducer.fit_transform(X_hybrid)

        # 5. HDBSCAN Clustering
        if self.use_gpu:
            # --- GPU HDBSCAN ---
            clusterer = CuHDBSCAN(
                min_cluster_size=self.min_samples,
                metric='euclidean',
                cluster_selection_epsilon=self.cluster_epsilon,
                # cuML HDBSCAN parameters are slightly different
                gen_min_span_tree=True
            )
            labels = clusterer.fit_predict(embedding_projection)
            # cuML returns probabilities_ as a property usually, but verify consistency
            if hasattr(clusterer, 'probabilities_'):
                probs = clusterer.probabilities_
            else:
                # Fallback if specific cuML version lacks probs
                probs = np.ones_like(labels, dtype=float)

            # Ensure we are back on CPU/Numpy for DataFrame operations
            if hasattr(labels, 'get'): labels = labels.get()  # cupy to numpy
            if hasattr(probs, 'get'): probs = probs.get()  # cupy to numpy

        else:
            # --- CPU HDBSCAN ---
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_samples,
                metric='euclidean',
                cluster_selection_epsilon=self.cluster_epsilon,
                prediction_data=True
            )
            labels = clusterer.fit_predict(embedding_projection)
            probs = clusterer.probabilities_

        # 6. Extract Interoperable Groups
        df_out = df.copy()
        df_out['cluster_label'] = labels
        df_out['cluster_prob'] = probs
        df_out['prediction'] = (labels != -1).astype(int)  # -1 is Noise

        groups = self._extract_groups(df_out, labels, probs)

        # 7. Evaluate
        metrics = self._evaluate(df_out)

        return ClusteringResult(groups=groups, clustered_df=df_out, metrics=metrics)

    def _extract_groups(self, df: pd.DataFrame, labels: np.ndarray, probs: np.ndarray) -> list[InteroperableGroup]:
        """Converts raw cluster labels into business logic objects."""
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        results = []
        for label in unique_labels:
            mask = (labels == label)
            cluster_rows = df[mask]

            # Metadata stats
            text_mode = cluster_rows[self.field_config.text].mode()
            description = text_mode[0] if not text_mode.empty else "Unknown"

            median_amount = float(cluster_rows[self.field_config.amount].median())

            # Cycle logic
            dates = pd.to_datetime(cluster_rows[self.field_config.date]).sort_values()
            if len(dates) > 1:
                diffs = dates.diff().dt.days.dropna()
                cycle_days = float(diffs.median())
            else:
                cycle_days = 0.0

            # Confidence is the mean probability of points belonging to this cluster
            confidence = float(probs[mask].mean())

            group = InteroperableGroup(
                group_id=f"cluster_{label}",
                description=description,
                amount_stats=median_amount,
                cycle_days=cycle_days,
                confidence=confidence,
                is_recurring=True  # By definition of being in a cluster
            )
            results.append(group)

        return results

    def _evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        """Calculates F1, Precision, Recall against the ground truth 'isRecurring'."""
        y_true = df[self.field_config.label].astype(int).values
        y_pred = df['prediction'].values

        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        return {
            "f1": float(f1),
            "precision": float(p),
            "recall": float(r)
        }

    def _empty_result(self, df: pd.DataFrame) -> ClusteringResult:
        """Fallback for when clustering cannot run."""
        df_out = df.copy()
        df_out['cluster_label'] = -1
        df_out['cluster_prob'] = 0.0
        df_out['prediction'] = 0

        metrics = self._evaluate(df_out)
        return ClusteringResult(groups=[], clustered_df=df_out, metrics=metrics)