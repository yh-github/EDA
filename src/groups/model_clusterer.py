import logging
import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass
from sklearn.metrics import precision_recall_fscore_support

# GPU/CPU Imports
try:
    import cupy as cp
    from cuml.manifold import UMAP as CuUMAP
    from cuml.cluster import HDBSCAN as CuHDBSCAN

    HAS_GPU_CLUSTERING = True
except ImportError:
    HAS_GPU_CLUSTERING = False

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
    def __init__(self, model: HybridModel, processor: HybridFeatureProcessor, min_samples: int = 2,
                 use_gpu: bool = True, voting_threshold: float = 0.5):
        self.model = model
        self.processor = processor
        self.min_samples = min_samples
        self.use_gpu = use_gpu and HAS_GPU_CLUSTERING
        self.voting_threshold = voting_threshold
        self.field_config = FieldConfig()
        self.model.eval()
        self.device = next(model.parameters()).device

    def cluster_features(self, feature_set: FeatureSet, df_original: pd.DataFrame) -> ModelClusteringResult:
        x_text = torch.from_numpy(feature_set.X_text).float().to(self.device)
        x_cont = torch.from_numpy(feature_set.X_continuous).float().to(self.device)
        x_cat = torch.from_numpy(feature_set.X_categorical).long().to(self.device)

        with torch.no_grad():
            latent_vectors = self.model.embed(x_text, x_cont, x_cat).cpu().numpy()
            logits = self.model.forward(x_text, x_cont, x_cat)
            probs = torch.sigmoid(logits).cpu().numpy()

        n_samples = len(df_original)
        if n_samples < self.min_samples:
            return self._empty_result(df_original)

        # UMAP
        if n_samples > 10:
            n_neighbors = min(15, n_samples - 1)
            if self.use_gpu:
                reducer = CuUMAP(n_neighbors=n_neighbors, n_components=5, min_dist=0.0, metric='cosine',
                                 random_state=42)
            else:
                reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=5, min_dist=0.0, metric='cosine',
                                    random_state=42, n_jobs=1)
            embedding_projection = reducer.fit_transform(latent_vectors)
        else:
            embedding_projection = latent_vectors

        # HDBSCAN
        effective_min = min(self.min_samples, n_samples)
        if self.use_gpu:
            clusterer = CuHDBSCAN(min_cluster_size=effective_min, metric='euclidean', cluster_selection_epsilon=0.0,
                                  gen_min_span_tree=True)
            labels = clusterer.fit_predict(embedding_projection)
            if hasattr(labels, 'get'): labels = labels.get()
        else:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=effective_min, metric='euclidean',
                                        cluster_selection_epsilon=0.0, prediction_data=True)
            labels = clusterer.fit_predict(embedding_projection)

        # Voting
        df_out = df_original.copy()
        df_out['cluster_label'] = labels
        df_out['prediction'] = 0

        unique_labels = set(labels)
        if -1 in unique_labels: unique_labels.remove(-1)

        for label in unique_labels:
            mask = (labels == label)
            if np.mean(probs[mask]) > self.voting_threshold:
                df_out.loc[mask, 'prediction'] = 1

        # Metrics
        y_true = df_out[self.field_config.label].astype(int).values
        y_pred = df_out['prediction'].values
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

        return ModelClusteringResult(df_out, {'precision': p, 'recall': r, 'f1': f1})

    def _empty_result(self, df):
        res = df.copy()
        res['prediction'] = 0
        return ModelClusteringResult(res, {'precision': 0, 'recall': 0, 'f1': 0})