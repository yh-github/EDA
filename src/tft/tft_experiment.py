import logging
import warnings
from pathlib import Path
from typing import Callable, Dict, Any

import lightning.pytorch as pl
import pandas as pd
import torch

from common.config import FieldConfig, EmbModel, ExperimentConfig
from common.data import create_train_val_test_split
from common.embedder import EmbeddingService
from common.feature_processor import FeatProcParams
from common.log_utils import setup_logging
# Default imports for the standard pipeline
from tft.tft_data import prepare_tft_data, build_tft_dataset
from tft.tft_runner import TFTRunner

logger = logging.getLogger("tft_experiment")


class TFTTuningExperiment:
    """
    A generalized experiment runner for TFT Hyperparameter tuning.
    Handles data loading, embedding, processing, and the Optuna tuning loop.
    """

    def __init__(
            self,
            study_name: str,
            best_model_path: str,
            min_encoder_len: int = 5,
            max_encoder_len: int = 150,
            batch_size: int = 2048,
            max_epochs: int = 10,
            n_trials: int = 20,
            search_space: dict[str, Any] | None = None,
            # dependency injection for data pipeline strategies
            prepare_data_fn: Callable = prepare_tft_data,
            build_dataset_fn: Callable = build_tft_dataset,
            data_path: str = "data/rec_data2.csv",
            log_dir: str = "logs/",
            use_aggregation: bool = False
    ):
        self.study_name = study_name
        self.best_model_path = best_model_path
        self.min_encoder_len = min_encoder_len
        self.max_encoder_len = max_encoder_len
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.n_trials = n_trials
        self.search_space = search_space
        self.prepare_data_fn = prepare_data_fn
        self.build_dataset_fn = build_dataset_fn
        self.data_path = data_path
        self.log_dir = Path(log_dir)
        self.use_aggregation = use_aggregation

    def run(self):
        setup_logging(self.log_dir, "tft_tuning")
        warnings.filterwarnings("ignore", category=FutureWarning)
        torch.set_float32_matmul_precision('medium')

        logger.info("Loading data...")
        field_config = FieldConfig()
        full_df = pd.read_csv(self.data_path).dropna(
            subset=[field_config.date, field_config.amount, field_config.text]
        )

        exp_params = ExperimentConfig()
        pl.seed_everything(exp_params.random_state, workers=True)

        # Split Data
        train_df, val_df, _ = create_train_val_test_split(
            test_size=0.2, val_size=0.2, full_df=full_df,
            random_state=exp_params.random_state
        )

        logger.info("Initializing Embedder...")
        emb_service = EmbeddingService(model_name=EmbModel.MPNET, max_length=64, batch_size=512)

        # Common feature params used by both experiments
        feat_params = FeatProcParams(
            use_is_positive=False, use_categorical_dates=True, use_cyclical_dates=True,
            use_continuous_amount=True, use_categorical_amount=False, k_top=0, n_bins=0
        )

        logger.info("Preparing Data...")
        # Use injected prepare function (Standard or Clustered)
        train_df_prepped, pca_model, processor, meta = self.prepare_data_fn(
            train_df, field_config, feat_params=feat_params,
            embedding_service=emb_service, fit_processor=True
        )
        val_df_prepped, _, _, _ = self.prepare_data_fn(
            val_df, field_config, embedding_service=emb_service,
            pca_model=pca_model, processor=processor, fit_processor=False
        )

        # Calculate class weights
        train_labels = train_df_prepped[field_config.label]
        n_pos = train_labels.sum()
        n_neg = len(train_labels) - n_pos
        pos_weight = float(n_neg / max(n_pos, 1))
        logger.info(f"Class Weight: {pos_weight:.2f}")

        # Build Datasets using injected build function
        train_ds = self.build_dataset_fn(
            train_df_prepped,
            field_config,
            meta,
            min_encoder_length=self.min_encoder_len,
            max_encoder_length=self.max_encoder_len
        )
        # Val DS must always be derived from Train DS to share scalers/encoders
        val_ds = train_ds.from_dataset(train_ds, val_df_prepped, predict=True, stop_randomization=True)

        def create_loader(is_train:bool):
            _ds = train_ds if is_train else val_ds
            return _ds.to_dataloader(
                train=is_train,
                batch_size=self.batch_size,
                num_workers=4,
                # DataLoader parameters
                pin_memory=True,
                persistent_workers=True
            )

        # Create Loaders
        train_loader = create_loader(True)
        val_loader = create_loader(False)

        # Run Tuning
        runner = TFTRunner(
            train_ds,
            train_loader,
            val_loader,
            # --- Pass the validation DataFrame explicitly ---
            val_df=val_df_prepped if self.use_aggregation else None,
            pos_weight=pos_weight,
            max_epochs=self.max_epochs,
            use_aggregation=self.use_aggregation
        )

        logger.info(f"Starting Study: {self.study_name}")
        study = runner.run_tuning(
            self.study_name,
            self.n_trials,
            exp_params.random_state,
            search_space=self.search_space,
            best_model_save_path=self.best_model_path
        )

        logger.info(f"BEST TRIAL F1: {study.best_value:.4f}")
        logger.info(f"Params: {study.best_params}")
        logger.info(f"Best model saved to: {self.best_model_path}")