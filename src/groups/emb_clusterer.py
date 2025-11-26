import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


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
class GroupCandidate:
    """Represents a raw cluster found by HDBSCAN, with features for classification."""
    group_id: str
    description: str
    # Features
    size: int
    confidence: float
    amt_median: float
    amt_cv: float  # Coeff of Variation
    days_median: float  # Median cycle
    days_std: float  # Cycle stability

    # Ground Truth (for training)
    true_label: int = 0  # 1 if majority of txns are recurring

    def to_feature_vector(self):
        """Returns the feature list for the ML model."""
        return [
            self.size,
            self.confidence,
            self.amt_median,
            self.amt_cv,
            self.days_median,
            self.days_std
        ]


class GroupClassifier:
    """
    A supervised model that learns to accept/reject candidate groups.
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight='balanced',
            random_state=42
        )
        self.feature_names = ['size', 'confidence', 'amt_median', 'amt_cv', 'days_median', 'days_std']

    def fit(self, candidates: list[GroupCandidate]):
        if not candidates:
            logger.warning("No candidates to fit.")
            return

        X = [c.to_feature_vector() for c in candidates]
        y = [c.true_label for c in candidates]

        self.model.fit(X, y)
        logger.info(f"Classifier trained on {len(X)} candidates. Classes: {self.model.classes_}")

        # Log feature importance
        imps = dict(zip(self.feature_names, self.model.feature_importances_))
        logger.info(f"Feature Importances: {imps}")

    def predict(self, candidate: GroupCandidate) -> bool:
        """Returns True if the candidate is predicted to be recurring."""
        vec = np.array([candidate.to_feature_vector()])
        return bool(self.model.predict(vec)[0] == 1)


class EmbClusterer:
    """
    Candidate Generator.
    Uses relaxed clustering to find ALL potential patterns.
    """

    def __init__(
            self,
            field_config: FieldConfig,
            feat_params: FeatProcParams,
            emb_service: EmbeddingService,
            # Relaxed Defaults for High Recall
            min_samples: int = 2,
            umap_components: int = 5,
            umap_neighbors: int = 15,
            cluster_epsilon: float = 0.0,
            use_gpu: bool = True,
            text_pca_components: int = 16
    ):
        self.field_config = field_config
        self.feat_params = feat_params
        self.emb_service = emb_service
        self.min_samples = min_samples
        self.umap_components = umap_components
        self.umap_neighbors = umap_neighbors
        self.cluster_epsilon = cluster_epsilon
        self.text_pca_components = text_pca_components

        self.use_gpu = use_gpu and HAS_GPU_CLUSTERING
        if use_gpu and not HAS_GPU_CLUSTERING:
            logger.warning("GPU clustering requested but cuML not found. Falling back to CPU.")

    def extract_candidates(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[GroupCandidate]]:
        """
        Runs clustering and returns (clustered_df, list_of_candidates).
        clustered_df has 'cluster_label' but NO 'prediction' yet.
        """
        if len(df) < self.min_samples:
            df_out = df.copy()
            df_out['cluster_label'] = -1
            return df_out, []

        # 1. Feature Gen
        processor = HybridFeatureProcessor(self.feat_params, self.field_config)
        meta = processor.fit(df)
        features_df = processor.transform(df)

        dense_cols = meta.continuous_scalable_cols + meta.cyclical_cols
        X_dense = features_df[dense_cols].values

        # 2. Text
        texts = df[self.field_config.text].tolist()
        X_text_raw = self.emb_service.embed(texts)

        n_samples = X_text_raw.shape[0]
        n_comps = min(self.text_pca_components, n_samples)
        if n_comps < 2:
            X_text_compressed = X_text_raw[:, :self.text_pca_components]
        else:
            pca = PCA(n_components=n_comps)
            X_text_compressed = pca.fit_transform(X_text_raw)

        # 3. Hybrid
        if X_dense.shape[1] > 0:
            scaler = StandardScaler()
            X_dense_scaled = scaler.fit_transform(X_dense)
            X_hybrid = np.hstack([X_text_compressed, X_dense_scaled])
        else:
            X_hybrid = X_text_compressed

        # 4. UMAP
        n_neighbors = min(self.umap_neighbors, len(df) - 1)
        if n_neighbors < 2:
            df_out = df.copy()
            df_out['cluster_label'] = -1
            return df_out, []

        if self.use_gpu:
            reducer = CuUMAP(n_neighbors=n_neighbors, n_components=self.umap_components, min_dist=0.0, metric='cosine',
                             random_state=42)
        else:
            reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=self.umap_components, min_dist=0.0,
                                metric='cosine', random_state=42)

        embedding_projection = reducer.fit_transform(X_hybrid)

        # 5. HDBSCAN
        if self.use_gpu:
            clusterer = CuHDBSCAN(min_cluster_size=self.min_samples, metric='euclidean',
                                  cluster_selection_epsilon=self.cluster_epsilon, gen_min_span_tree=True)
            labels = clusterer.fit_predict(embedding_projection)
            probs = getattr(clusterer, 'probabilities_', np.ones_like(labels))
            if hasattr(labels, 'get'): labels = labels.get()
            if hasattr(probs, 'get'): probs = probs.get()
        else:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_samples, metric='euclidean',
                                        cluster_selection_epsilon=self.cluster_epsilon, prediction_data=True)
            labels = clusterer.fit_predict(embedding_projection)
            probs = clusterer.probabilities_

        # 6. Build Candidates
        df_out = df.copy()
        df_out['cluster_label'] = labels
        df_out['cluster_prob'] = probs

        candidates = self._build_candidates(df_out, labels, probs)
        return df_out, candidates

    def _build_candidates(self, df: pd.DataFrame, labels: np.ndarray, probs: np.ndarray) -> list[GroupCandidate]:
        unique_labels = set(labels)
        if -1 in unique_labels: unique_labels.remove(-1)

        candidates = []
        for label in unique_labels:
            mask = (labels == label)
            cluster_rows = df[mask]

            # Text
            text_mode = cluster_rows[self.field_config.text].mode()
            description = text_mode[0] if not text_mode.empty else "Unknown"

            # Amount Stats
            amounts = cluster_rows[self.field_config.amount]
            amt_median = float(amounts.median())
            amt_mean_abs = amounts.abs().mean()
            amt_cv = amounts.std() / amt_mean_abs if amt_mean_abs > 1e-3 else 0.0

            # Cycle Stats
            dates = pd.to_datetime(cluster_rows[self.field_config.date]).sort_values()
            cycle_days = 0.0
            days_std = -1.0

            if len(dates) > 1:
                diffs = dates.diff().dt.days.dropna()
                cycle_days = float(diffs.median())
                if len(diffs) > 1:
                    days_std = float(diffs.std())
                else:
                    days_std = 0.0  # Only 2 points = perfectly stable interval

            confidence = float(probs[mask].mean())
            size = int(mask.sum())

            # Ground Truth Logic
            # Check if majority of rows are actually recurring
            true_labels = cluster_rows[self.field_config.label].astype(int)
            is_recurring_majority = (true_labels.sum() / len(true_labels)) > 0.5

            cand = GroupCandidate(
                group_id=f"cluster_{label}",
                description=description,
                size=size,
                confidence=confidence,
                amt_median=amt_median,
                amt_cv=amt_cv,
                days_median=cycle_days,
                days_std=days_std,
                true_label=1 if is_recurring_majority else 0
            )
            candidates.append(cand)

        return candidates