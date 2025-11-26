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
from common.feature_processor import HybridFeatureProcessor
from common.data import FeatureSet
from pointwise.classifier import HybridModel

logger = logging.getLogger(__name__)


@dataclass
class ModelClusteringResult:
    clustered_df: pd.DataFrame
    metrics: dict[str, float]


class ModelBasedClusterer:
    """
    Clusters transactions using the Latent Space of a pre-trained HybridModel,
    then VERIFIES clusters using the model's own probability scores.
    """

    def __init__(
            self,
            model: HybridModel,
            processor: HybridFeatureProcessor,
            min_samples: int = 2,
            use_gpu: bool = True,
            # New Threshold: A cluster must have this avg probability to be kept
            conf_threshold: float = 0.5
    ):
        self.model = model
        self.processor = processor
        self.min_samples = min_samples
        self.use_gpu = use_gpu and HAS_GPU_CLUSTERING
        self.conf_threshold = conf_threshold
        self.field_config = FieldConfig()

        self.model.eval()
        # Detect device from model parameters
        self.device = next(model.parameters()).device

    def cluster_features(self, feature_set: FeatureSet, df_original: pd.DataFrame) -> ModelClusteringResult:
        """
        Runs clustering on prepared features.
        """
        # Move to GPU/Tensor
        x_text = torch.from_numpy(feature_set.X_text).float().to(self.device)
        x_cont = torch.from_numpy(feature_set.X_continuous).float().to(self.device)
        x_cat = torch.from_numpy(feature_set.X_categorical).long().to(self.device)

        # --- 1. Get Latent Embeddings AND Probabilities ---
        with torch.no_grad():
            # Embeddings for Clustering
            latent_vectors = self.model.embed(x_text, x_cont, x_cat).cpu().numpy()

            # Logits for Verification
            logits = self.model.head(torch.tensor(latent_vectors).to(self.device))  # Re-use latent to save compute
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()

        n_samples = len(df_original)

        # Default: No clusters
        labels = np.full(n_samples, -1)

        # --- 2. UMAP + HDBSCAN (Structure Discovery) ---
        if n_samples >= self.min_samples:
            # Skip Dim Reduction for tiny datasets (< 10 samples)
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
                embedding_projection = latent_vectors

            # HDBSCAN
            effective_min_samples = min(self.min_samples, n_samples)
            if self.use_gpu:
                clusterer = CuHDBSCAN(min_cluster_size=effective_min_samples, metric='euclidean',
                                      cluster_selection_epsilon=0.0, gen_min_span_tree=True)
                labels = clusterer.fit_predict(embedding_projection)
                if hasattr(labels, 'get'): labels = labels.get()
            else:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=effective_min_samples, metric='euclidean',
                                            cluster_selection_epsilon=0.0, prediction_data=True)
                labels = clusterer.fit_predict(embedding_projection)

        # --- 3. Prediction Logic (Cluster-then-Verify) ---
        df_out = df_original.copy()
        df_out['cluster_label'] = labels
        df_out['model_prob'] = probs
        df_out['prediction'] = 0  # Default 0

        unique_labels = set(labels)
        if -1 in unique_labels: unique_labels.remove(-1)

        for label in unique_labels:
            mask = (labels == label)

            # Calculate Cluster Confidence
            # (Average probability of transactions in this cluster)
            avg_cluster_prob = probs[mask].mean()

            # If the model generally thinks this group is recurring, accept it.
            # Otherwise, it's just a cluster of non-recurring items (e.g. Coffee).
            if avg_cluster_prob >= self.conf_threshold:
                df_out.loc[mask, 'prediction'] = 1

        # --- 4. Evaluate ---
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