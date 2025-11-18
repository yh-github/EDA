import logging
import json
import pandas as pd
import numpy as np
import diskcache
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score, precision_score, recall_score

from config import FieldConfig, FilterConfig, EmbModel
from embedder import EmbeddingService
from greedy_analyzer import GreedyGroupAnalyzer, GroupStabilityStatus
from hyper_tuner import HyperTuner
from log_utils import setup_logging

logger = logging.getLogger(__name__)


class GreedyTuner:
    def __init__(self, ind:int, filter_direction:int):
        self.cache_dir = Path(f"cache/greedy_results/{ind}")
        self.cache = diskcache.Cache(str(self.cache_dir))
        self.field_config = FieldConfig()
        self.filter_direction = filter_direction
        self.accounts: pd.DataFrame|None = None
        self.df: pd.DataFrame|None = None

    def load_data(self):
        logger.info("Loading data for tuning...")
        df_train_val, df_cleaned = HyperTuner.load_and_split_data(
            filter_direction=self.filter_direction,
            data_path=Path('data/rec_data2.csv')
        )
        self.df = df_train_val
        self.accounts = self.df[self.field_config.accountId].unique()
        logger.info(f"Loaded {len(self.df)} rows, {len(self.accounts)} accounts.")

    def precompute_embeddings(self, model_name=EmbModel.MPNET):
        """
        Pre-computes embeddings for all texts so the inner loop is fast.
        """
        logger.info(f"Pre-computing embeddings using {model_name}...")
        self.emb_service = EmbeddingService(model_name=model_name, max_length=64, batch_size=256)
        # This will populate the disk cache in EmbeddingService
        self.emb_service.embed(self.df[self.field_config.text].unique().tolist())

    def run_grid(self, grid_config: dict):
        param_list = list(ParameterGrid(grid_config))
        logger.info(f"Starting Grid Search with {len(param_list)} combinations.")

        best_f1 = -1.0
        best_params = None

        for i, params in enumerate(param_list):
            # Check cache first
            param_key = json.dumps(params, sort_keys=True)
            if param_key in self.cache:
                metrics = self.cache[param_key]
                logger.info(f"[{i + 1}/{len(param_list)}] CACHED: F1={metrics['f1']:.4f} | {params}")
            else:
                # Run Evaluation
                metrics = self._evaluate_params(params)
                self.cache[param_key] = metrics
                logger.info(
                    f"[{i + 1}/{len(param_list)}] RUN: F1={metrics['f1']:.4f} (P={metrics['precision']:.3f}, R={metrics['recall']:.3f}) | {params}")

            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_params = params

        logger.info("=" * 50)
        logger.info(f"BEST F1: {best_f1:.4f}")
        logger.info(f"PARAMS: {best_params}")
        logger.info("=" * 50)

    def _evaluate_params(self, params) -> dict:
        # Construct Configs from Params
        filter_config = FilterConfig(
            min_txns_for_period=params.get('min_txns', 3),
            date_std_threshold=params.get('date_std', 2.0)
        )

        analyzer = GreedyGroupAnalyzer(
            filter_config=filter_config,
            field_config=self.field_config,
            sim_threshold=params.get('sim_threshold', 0.90),
            amount_tol_abs=params.get('amount_tol_abs', 2.00),
            amount_tol_pct=params.get('amount_tol_pct', 0.05)
        )

        all_true = []
        all_pred = []

        # Run on all accounts
        # Note: For speed in tuning, you might want to sample accounts (e.g. first 200)
        # accounts_to_run = self.accounts[:200]
        accounts_to_run = self.accounts

        for acc_id in accounts_to_run:
            acc_df = self.df[self.df[self.field_config.accountId] == acc_id]
            if acc_df.empty: continue

            # Get embeddings (Cached lookups)
            embeddings = self.emb_service.embed(acc_df[self.field_config.text].tolist())

            # Analyze
            groups = analyzer.analyze_account(acc_df, embeddings)

            # Map back to labels
            pred_labels = np.zeros(len(acc_df))
            acc_ids = acc_df[self.field_config.trId].values

            for grp in groups:
                if grp.status == GroupStabilityStatus.STABLE:
                    # Find indices of these transaction IDs
                    mask = np.isin(acc_ids, grp.transaction_ids)
                    pred_labels[mask] = 1

            all_true.extend(acc_df[self.field_config.label].values)
            all_pred.extend(pred_labels)

        return {
            'f1': f1_score(all_true, all_pred),
            'precision': precision_score(all_true, all_pred),
            'recall': recall_score(all_true, all_pred)
        }