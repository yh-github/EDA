import logging
import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass
from sklearn.metrics import precision_recall_fscore_support

# --- GPU Imports ---
try:
    import cupy as cp
    from cuml.manifold import UMAP as CuUMAP
    from cuml.cluster import HDBSCAN as CuHDBSCAN

    HAS_GPU_CLUSTERING = True
except ImportError:
    HAS_GPU_CLUSTERING = False

# --- CPU Imports ---
try:
    import umap
    import hdbscan
except ImportError:
    import umap.umap_ as umap
    import hdbscan

from common.config import FieldConfig
from common.feature_processor import HybridFeatureProcessor, FeatProcParams
from common.data import FeatureSet, TransactionDataset
from pointwise.classifier import HybridModel

logger = logging.getLogger(__name__)


@dataclass
class ModelClusteringResult:
    clustered_df: pd.DataFrame
    metrics: dict[str, float]


class ModelBasedClusterer:
    """
    Clusters transactions using the Latent Space of a pre-trained HybridModel.
    """

    def __init__(
            self,
            model: HybridModel,
            processor: HybridFeatureProcessor,
            min_samples: int = 2,
            use_gpu: bool = True
    ):
        self.model = model
        self.processor = processor
        self.min_samples = min_samples
        self.use_gpu = use_gpu and HAS_GPU_CLUSTERING
        self.field_config = FieldConfig()

        self.model.eval()
        self.device = next(model.parameters()).device

    def cluster_and_evaluate(self, df: pd.DataFrame, set_name: str) -> ModelClusteringResult:
        raise NotImplementedError("Use 'cluster_features' method with prepared tensors.")

    def cluster_features(self, feature_set: FeatureSet, df_original: pd.DataFrame) -> ModelClusteringResult:
        """
        Runs clustering on prepared features.
        """
        # Move to GPU/Tensor
        x_text = torch.from_numpy(feature_set.X_text).float().to(self.device)
        x_cont = torch.from_numpy(feature_set.X_continuous).float().to(self.device)
        x_cat = torch.from_numpy(feature_set.X_categorical).long().to(self.device)

        # --- 1. Get Latent Embeddings ---
        with torch.no_grad():
            # calling model.embed() gives the penultimate layer
            latent_vectors = self.model.embed(x_text, x_cont, x_cat).cpu().numpy()

        n_samples = len(df_original)
        if n_samples < self.min_samples:
            return self._empty_result(df_original)

        # --- 2. UMAP Projection (Conditional) ---
        # FIX: UMAP requires n_samples > n_components (5).
        # If samples are small (<= 10), we skip UMAP and cluster the latent vectors directly.
        if n_samples > 10:
            n_neighbors = min(15, n_samples - 1)

            if self.use_gpu:
                reducer = CuUMAP(n_neighbors=n_neighbors, n_components=5, min_dist=0.0, metric='cosine',
                                 random_state=42)
                embedding_projection = reducer.fit_transform(latent_vectors)
            else:
                reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=5, min_dist=0.0, metric='cosine',
                                    random_state=42, n_jobs=1)
                embedding_projection = reducer.fit_transform(latent_vectors)
        else:
            # Skip Dim Reduction for tiny datasets
            embedding_projection = latent_vectors

        # --- 3. HDBSCAN Clustering ---
        # Ensure min_cluster_size isn't larger than the dataset itself
        effective_min_samples = min(self.min_samples, n_samples)

        if self.use_gpu:
            clusterer = CuHDBSCAN(
                min_cluster_size=effective_min_samples,
                metric='euclidean',
                cluster_selection_epsilon=0.0,
                gen_min_span_tree=True
            )
            labels = clusterer.fit_predict(embedding_projection)

            if hasattr(labels, 'get'): labels = labels.get()
        else:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=effective_min_samples,
                metric='euclidean',
                cluster_selection_epsilon=0.0,
                prediction_data=True
            )
            labels = clusterer.fit_predict(embedding_projection)

        # --- 4. Prediction Logic ---
        # Anyone in a cluster is "Recurring"
        # Anyone in Noise (-1) is "Non-Recurring"
        df_out = df_original.copy()
        df_out['cluster_label'] = labels
        df_out['prediction'] = (labels != -1).astype(int)

        # --- 5. Evaluate ---
        y_true = df_out[self.field_config.label].astype(int).values
        y_pred = df_out['prediction'].values

        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

        return ModelClusteringResult(
            clustered_df=df_out,
            metrics={'precision': p, 'recall': r, 'f1': f1}
        )

    def _empty_result(self, df):
        res = df.copy()
        res['prediction'] = 0
        return ModelClusteringResult(res, {'precision': 0, 'recall': 0, 'f1': 0})