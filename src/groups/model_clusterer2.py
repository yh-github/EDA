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
from tft.group import InteroperableGroup

logger = logging.getLogger(__name__)


@dataclass
class ModelClusteringResult:
    clustered_df: pd.DataFrame
    groups: list[InteroperableGroup]
    metrics: dict[str, float]


class ModelBasedClusterer:
    def __init__(
            self,
            model: HybridModel,
            processor: HybridFeatureProcessor,
            min_samples: int = 2,
            use_gpu: bool = True,
            # --- Logic Gates ---
            voting_threshold: float = 0.4,
            anchoring_threshold: float = 0.85,
            max_amount_cv: float = 0.2,
            min_cycle_days: float = 6.0,
            max_period_std: float = 4.0,
            # --- Recovery Params ---
            enable_recovery: bool = True,
            recovery_distance_threshold: float = 2.0  # Euclidean dist in Latent Space (64-dim)
    ):
        self.model = model
        self.processor = processor
        self.min_samples = min_samples
        self.use_gpu = use_gpu and HAS_GPU_CLUSTERING

        self.voting_threshold = voting_threshold
        self.anchoring_threshold = anchoring_threshold
        self.max_amount_cv = max_amount_cv
        self.min_cycle_days = min_cycle_days
        self.max_period_std = max_period_std
        self.enable_recovery = enable_recovery
        self.recovery_distance_threshold = recovery_distance_threshold

        self.field_config = FieldConfig()
        self.model.eval()
        self.device = next(model.parameters()).device

    def cluster_features(self, feature_set: FeatureSet, df_original: pd.DataFrame) -> ModelClusteringResult:
        x_text = torch.from_numpy(feature_set.X_text).float().to(self.device)
        x_cont = torch.from_numpy(feature_set.X_continuous).float().to(self.device)
        x_cat = torch.from_numpy(feature_set.X_categorical).long().to(self.device)

        # --- 1. Inference ---
        with torch.no_grad():
            # Latent vectors (64-dim) are the "Truth" of the model
            latent_vectors = self.model.embed(x_text, x_cont, x_cat).cpu().numpy()
            logits = self.model.forward(x_text, x_cont, x_cat)
            probs = torch.sigmoid(logits).cpu().numpy()

        n_samples = len(df_original)
        if n_samples < self.min_samples:
            return self._empty_result(df_original)

        # --- 2. UMAP ---
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

        # --- 3. HDBSCAN ---
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

        # --- 4. Validation & Filtering ---
        df_out = df_original.copy()
        df_out['cluster_label'] = labels
        df_out['pointwise_prob'] = probs
        df_out['prediction'] = 0

        unique_labels = set(labels)
        if -1 in unique_labels: unique_labels.remove(-1)

        valid_groups_meta = {}  # Store metadata for recovery

        for label in unique_labels:
            mask = (labels == label)
            cluster_rows = df_out[mask]
            cluster_probs = probs[mask]

            # Vote
            model_vote_pass = (np.mean(cluster_probs) > self.voting_threshold) or \
                              (np.max(cluster_probs) > self.anchoring_threshold)

            if not model_vote_pass: continue

            # Logic Stats
            stats = self._analyze_cluster(cluster_rows)

            # Filters
            if stats['amt_cv'] > self.max_amount_cv: continue
            if stats['cycle_days'] < self.min_cycle_days: continue
            if stats['period_std'] > self.max_period_std: continue

            # Accept
            df_out.loc[mask, 'prediction'] = 1

            # Store for Recovery Phase
            # We need the centroid of the LATENT VECTORS, not the UMAP projection
            cluster_latent = latent_vectors[mask]
            centroid = np.mean(cluster_latent, axis=0)

            group_obj = InteroperableGroup(
                group_id=f"grp_{label}",
                description=stats['description'],
                amount_stats=stats['amt_median'],
                cycle_days=stats['cycle_days'],
                confidence=float(np.mean(cluster_probs)),
                is_recurring=True
            )
            valid_groups_meta[label] = {
                "meta": group_obj,
                "centroid": centroid,
                "indices": np.where(mask)[0]  # Row indices
            }

        # --- 5. Recovery Phase (The Recall Booster) ---
        if self.enable_recovery and len(valid_groups_meta) > 0:
            self._recover_missed_transactions(
                df_out, latent_vectors, valid_groups_meta
            )

        # --- 6. Final Groups List & Eval ---
        final_groups = [v["meta"] for v in valid_groups_meta.values()]

        y_true = df_out[self.field_config.label].astype(int).values
        y_pred = df_out['prediction'].values
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

        return ModelClusteringResult(df_out, final_groups, {'precision': p, 'recall': r, 'f1': f1})

    def _recover_missed_transactions(self, df_out: pd.DataFrame, latent_vectors: np.ndarray, valid_groups: dict):
        """
        Scans rejected transactions (Noise or Rejected Clusters).
        If they match a Valid Cluster (Distance + Amount), pull them in.
        """
        # Find candidates: Prediction is 0
        candidate_mask = (df_out['prediction'] == 0)
        if not candidate_mask.any():
            return

        cand_indices = np.where(candidate_mask)[0]
        cand_vectors = latent_vectors[cand_indices]
        cand_rows = df_out.iloc[cand_indices]

        # Iterate over valid groups "Magnets"
        for label, data in valid_groups.items():
            centroid = data['centroid']
            target_amt = data['meta'].amount_stats

            # 1. Vector Distance Check (Batch)
            # Calculate Euclidean dist from all candidates to this centroid
            dists = np.linalg.norm(cand_vectors - centroid, axis=1)

            # 2. Amount Check (Batch)
            # Allow 10% variance or $2.00, whichever is larger
            # (Use simple pandas/numpy vectorized ops)
            cand_amts = cand_rows[self.field_config.amount].values
            amt_diff = np.abs(cand_amts - target_amt)
            amt_pass = (amt_diff < 2.0) | (amt_diff < (np.abs(target_amt) * 0.1))

            # 3. Combine
            # Distance must be close AND Amount must match
            # Threshold depends on latent space scale.
            # Since we used LayerNorm(64) in model, vectors are somewhat normalized.
            # A threshold of ~2.0-3.0 usually captures "Same Merchant".
            matches = (dists < self.recovery_distance_threshold) & amt_pass

            if matches.any():
                # Get indices of matched candidates (subset -> original)
                recovered_indices = cand_indices[matches]

                # Update DataFrame
                df_out.iloc[recovered_indices, df_out.columns.get_loc('prediction')] = 1
                # Optionally update cluster label to match the magnet
                df_out.iloc[recovered_indices, df_out.columns.get_loc('cluster_label')] = label

                # Note: We are not currently checking Dates/Cycle here for speed.
                # Assumption: If Model(Text) says same AND Amount says same,
                # it's extremely likely to be the same subscription, even if the date is off-cycle.

    def _analyze_cluster(self, df_cluster: pd.DataFrame) -> dict:
        text_mode = df_cluster[self.field_config.text].mode()
        desc = text_mode[0] if not text_mode.empty else "Unknown"

        amts = df_cluster[self.field_config.amount]
        amt_median = float(amts.median())
        amt_mean_abs = amts.abs().mean()
        amt_cv = (amts.std() / amt_mean_abs) if amt_mean_abs > 1e-3 else 0.0

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
