import logging
import json
import pandas as pd
import numpy as np
import diskcache
from pathlib import Path
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score, precision_score, recall_score

from config import FieldConfig, FilterConfig, ExperimentConfig, EmbModel, ClusteringStrategy
from embedder import EmbeddingService
from unified_analyzer import StrategyDispatcher
from group_analyzer import GroupStabilityStatus
from hyper_tuner import HyperTuner  # reused for data loading

logger = logging.getLogger(__name__)


class ClusteringExperiment:
    """
    Runs a grid search over unsupervised clustering strategies (Greedy vs DBSCAN)
    evaluated against ground truth labels.
    """

    def __init__(self, experiment_name: str, filter_direction: int):
        self.cache_dir = Path(f"cache/clustering_exp/{experiment_name}")
        self.cache = diskcache.Cache(str(self.cache_dir))

        self.field_config = FieldConfig()
        self.exp_config = ExperimentConfig()
        self.filter_direction = filter_direction

        self.df = None
        self.accounts = None
        self.emb_service = None

    def load_data(self):
        """Loads and filters data, same as HyperTuner."""
        logger.info("Loading data for clustering experiment...")
        df_train_val, _ = HyperTuner.load_and_split_data(
            filter_direction=self.filter_direction,
            data_path=Path('data/rec_data2.csv'),
            field_config=self.field_config,
            random_state=self.exp_config.random_state
        )
        self.df = df_train_val
        self.accounts = self.df[self.field_config.accountId].unique()
        logger.info(f"Loaded {len(self.df)} rows, {len(self.accounts)} accounts.")

    def precompute_embeddings(self, model_name=EmbModel.MPNET):
        """Pre-computes embeddings so we don't re-run BERT during the grid search."""
        logger.info(f"Pre-computing embeddings using {model_name}...")
        self.emb_service = EmbeddingService(model_name=model_name, max_length=64, batch_size=256)
        all_texts = self.df[self.field_config.text].unique().tolist()
        self.emb_service.embed(all_texts)  # Populates internal cache

    def run_grid(self, grid_config: dict):
        param_list = list(ParameterGrid(grid_config))
        logger.info(f"Starting Experiment with {len(param_list)} configurations.")

        results = []

        for i, params in enumerate(param_list):
            # Create a stable cache key
            param_key = json.dumps(params, sort_keys=True)

            if param_key in self.cache:
                metrics = self.cache[param_key]
                logger.info(f"[{i + 1}/{len(param_list)}] CACHED | F1: {metrics['f1']:.4f} | {params}")
            else:
                metrics = self._evaluate_single_run(params)
                self.cache[param_key] = metrics
                logger.info(f"[{i + 1}/{len(param_list)}] DONE | F1: {metrics['f1']:.4f} | {params}")

            results.append({**params, **metrics})

        # Summary
        res_df = pd.DataFrame(results)
        best_run = res_df.loc[res_df['f1'].idxmax()]

        logger.info("=" * 60)
        logger.info(f"BEST RUN F1: {best_run['f1']:.4f}")
        logger.info(f"Strategy: {best_run.get('strategy')}")
        logger.info("=" * 60)
        return res_df

    def _evaluate_single_run(self, params: dict) -> dict:
        # 1. Map Grid Params -> FilterConfig
        config = FilterConfig(
            strategy=ClusteringStrategy(params.get('strategy', 'greedy')),

            # Shared
            stability_metric=params.get('stability_metric', 'std'),
            date_variance_threshold=params.get('date_var', 2.0),
            amount_variance_threshold=params.get('amt_var', 1.0),

            # Greedy/Lexical
            greedy_sim_threshold=params.get('greedy_sim', 0.90),
            lexical_sim_threshold=params.get('lexical_sim', 0.85),

            # DBSCAN
            dbscan_eps=params.get('dbscan_eps', 0.5)
        )

        # 2. Instantiate
        analyzer = StrategyDispatcher(config, self.field_config)

        all_true = []
        all_pred = []

        for acc_id in self.accounts:
            mask = (self.df[self.field_config.accountId] == acc_id)
            acc_df = self.df[mask]
            if acc_df.empty: continue

            # If strategy is Lexical, we don't strictly need embeddings,
            # but passing them as None is handled by the analyzer.
            # If Greedy/DBSCAN, we slice the matrix.
            if config.strategy == 'lexical':
                embeddings = None
            else:
                embeddings = self.all_embeddings[mask]

            groups = analyzer.analyze_account(acc_df, embeddings)

            # Map back to labels
            pred_labels = np.zeros(len(acc_df), dtype=int)
            tx_ids = acc_df[self.field_config.trId].values

            for grp in groups:
                if grp.status == GroupStabilityStatus.STABLE:
                    txn_mask = np.isin(tx_ids, grp.transaction_ids)
                    pred_labels[txn_mask] = 1

            all_true.extend(acc_df[self.field_config.label].values)
            all_pred.extend(pred_labels)

        metrics = {
            'f1': f1_score(all_true, all_pred),
            'precision': precision_score(all_true, all_pred, zero_division=0),
            'recall': recall_score(all_true, all_pred, zero_division=0)
        }

        # ADDED LOGGING HERE
        logger.info(
            f"RUN RESULTS | F1: {metrics['f1']:.4f} | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f}")

        return metrics