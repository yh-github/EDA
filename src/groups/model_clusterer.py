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
# Reuse the InteroperableGroup dataclass for standardized output
from tft.group import InteroperableGroup

logger = logging.getLogger(__name__)


@dataclass
class ModelClusteringResult:
    clustered_df: pd.DataFrame
    groups: list[InteroperableGroup]  # Now returning rich metadata
    metrics: dict[str, float]


class ModelBasedClusterer:
    def __init__(
            self,
            model: HybridModel,
            processor: HybridFeatureProcessor,
            min_samples: int = 2,
            use_gpu: bool = True,
            # --- Logic Gates ---
            voting_threshold: float = 0.4,  # Model must be 40% sure on average
            anchoring_threshold: float = 0.85,  # OR at least one txn is 85% sure
            max_amount_cv: float = 0.2,  # Allow 20% variance in price (e.g. utility bills)
            min_cycle_days: float = 6.0,  # Minimum cycle (approx weekly)
            max_period_std: float = 4.0  # Allow +/- 4 days jitter in the cycle
    ):
        self.model = model
        self.processor = processor
        self.min_samples = min_samples
        self.use_gpu = use_gpu and HAS_GPU_CLUSTERING

        # Logic Config
        self.voting_threshold = voting_threshold
        self.anchoring_threshold = anchoring_threshold
        self.max_amount_cv = max_amount_cv
        self.min_cycle_days = min_cycle_days
        self.max_period_std = max_period_std

        self.field_config = FieldConfig()
        self.model.eval()
        self.device = next(model.parameters()).device

    def cluster_features(self, feature_set: FeatureSet, df_original: pd.DataFrame) -> ModelClusteringResult:
        x_text = torch.from_numpy(feature_set.X_text).float().to(self.device)
        x_cont = torch.from_numpy(feature_set.X_continuous).float().to(self.device)
        x_cat = torch.from_numpy(feature_set.X_categorical).long().to(self.device)

        # --- 1. Inference ---
        with torch.no_grad():
            latent_vectors = self.model.embed(x_text, x_cont, x_cat).cpu().numpy()
            logits = self.model.forward(x_text, x_cont, x_cat)
            probs = torch.sigmoid(logits).cpu().numpy()

        n_samples = len(df_original)
        if n_samples < self.min_samples:
            return self._empty_result(df_original)

        # --- 2. Dimensionality Reduction (UMAP) ---
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

        # --- 3. Density Clustering (HDBSCAN) ---
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

        # --- 4. Validation & Logic ---
        df_out = df_original.copy()
        df_out['cluster_label'] = labels
        df_out['pointwise_prob'] = probs
        df_out['prediction'] = 0

        unique_labels = set(labels)
        if -1 in unique_labels: unique_labels.remove(-1)

        valid_groups = []

        for label in unique_labels:
            mask = (labels == label)
            cluster_rows = df_out[mask]
            cluster_probs = probs[mask]

            # A. Model Voting (The "Intuition")
            # Pass if Average > Threshold OR Max > Anchor
            model_vote_pass = (np.mean(cluster_probs) > self.voting_threshold) or \
                              (np.max(cluster_probs) > self.anchoring_threshold)

            if not model_vote_pass:
                continue

            # B. Structure Analysis (The "Physics")
            stats = self._analyze_cluster(cluster_rows)

            # C. Business Logic Filters
            # Reject High Variance Amounts (unless very confident model vote?) -> Enforce for now
            if stats['amt_cv'] > self.max_amount_cv:
                continue

            # Reject Bursts (Daily txns)
            if stats['cycle_days'] < self.min_cycle_days:
                continue

            # Reject Irregular Periods (High Jitter)
            # Note: We allow '0' std which means perfect regularity (or only 2 txns)
            if stats['period_std'] > self.max_period_std:
                continue

            # --- ACCEPT ---
            df_out.loc[mask, 'prediction'] = 1

            # Create Metadata Object
            group_obj = InteroperableGroup(
                group_id=f"grp_{label}",
                description=stats['description'],
                amount_stats=stats['amt_median'],
                cycle_days=stats['cycle_days'],
                confidence=float(np.mean(cluster_probs)),
                is_recurring=True
            )
            valid_groups.append(group_obj)

        # --- 5. Evaluate ---
        y_true = df_out[self.field_config.label].astype(int).values
        y_pred = df_out['prediction'].values
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

        return ModelClusteringResult(df_out, valid_groups, {'precision': p, 'recall': r, 'f1': f1})

    def _analyze_cluster(self, df_cluster: pd.DataFrame) -> dict:
        """Calculates physical properties of the cluster."""
        # Text
        text_mode = df_cluster[self.field_config.text].mode()
        desc = text_mode[0] if not text_mode.empty else "Unknown"

        # Amount Stats
        amts = df_cluster[self.field_config.amount]
        amt_median = float(amts.median())
        amt_mean_abs = amts.abs().mean()
        # CV = Std / Mean
        amt_cv = (amts.std() / amt_mean_abs) if amt_mean_abs > 1e-3 else 0.0

        # Cycle Stats
        dates = pd.to_datetime(df_cluster[self.field_config.date]).sort_values()
        cycle_days = 0.0
        period_std = 0.0

        if len(dates) > 1:
            diffs = dates.diff().dt.days.dropna()
            cycle_days = float(diffs.median())
            if len(diffs) > 1:
                period_std = float(diffs.std())

        return {
            "description": desc,
            "amt_median": amt_median,
            "amt_cv": amt_cv,
            "cycle_days": cycle_days,
            "period_std": period_std
        }

    def _empty_result(self, df):
        res = df.copy()
        res['prediction'] = 0
        return ModelClusteringResult(res, [], {'precision': 0, 'recall': 0, 'f1': 0})