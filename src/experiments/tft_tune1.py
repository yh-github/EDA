import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import lightning.pytorch as pl
import pandas as pd
import torch

from common.config import FieldConfig, EmbModel, ExperimentConfig
from common.data import create_train_val_test_split
from common.log_utils import setup_logging
from tft.tft_data import build_tft_dataset, prepare_tft_data
from common.feature_processor import FeatProcParams
from common.embedder import EmbeddingService
from tft.tft_runner import TFTRunner

setup_logging(Path("logs/"), "tft_tuning")
logger = logging.getLogger("tft_tuner")
torch.set_float32_matmul_precision('medium')

MAX_ENCODER_LEN = 150
BATCH_SIZE = 2048
MAX_EPOCHS = 20
N_TRIALS = 30
STUDY_NAME = "tft_optimization_outgoing"
BEST_MODEL_PATH = "cache/tft_models/best_tune1_model.pt"

if __name__ == "__main__":
    logger.info("Loading data...")
    field_config = FieldConfig()
    full_df = pd.read_csv("data/rec_data2.csv").dropna(
        subset=[field_config.date, field_config.amount, field_config.text])

    exp_params = ExperimentConfig()
    pl.seed_everything(exp_params.random_state, workers=True)

    train_df, val_df, _ = create_train_val_test_split(test_size=0.2, val_size=0.2, full_df=full_df,
                                                      random_state=exp_params.random_state)

    emb_service = EmbeddingService(model_name=EmbModel.MPNET, max_length=64, batch_size=512)
    feat_params = FeatProcParams(
        use_is_positive=False, use_categorical_dates=True, use_cyclical_dates=True,
        use_continuous_amount=True, use_categorical_amount=False, k_top=0, n_bins=0
    )

    train_df_prepped, pca_model, processor, meta = prepare_tft_data(
        train_df, field_config, feat_params, emb_service, fit_processor=True
    )
    val_df_prepped, _, _, _ = prepare_tft_data(
        val_df, field_config, embedding_service=emb_service,
        pca_model=pca_model, processor=processor, fit_processor=False
    )

    train_labels = train_df_prepped[field_config.label]
    pos_weight = float((len(train_labels) - train_labels.sum()) / max(train_labels.sum(), 1))
    logger.info(f"Class Weight: {pos_weight:.2f}")

    train_ds = build_tft_dataset(train_df_prepped, field_config, meta, max_encoder_length=MAX_ENCODER_LEN)
    val_ds = train_ds.from_dataset(train_ds, val_df_prepped, predict=True, stop_randomization=True)

    train_loader = train_ds.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=12, pin_memory=True)
    val_loader = val_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=12, pin_memory=True)

    runner = TFTRunner(train_ds, train_loader, val_loader, pos_weight=pos_weight, max_epochs=MAX_EPOCHS)

    logger.info(f"Starting Study: {STUDY_NAME}")
    study = runner.run_tuning(
        STUDY_NAME,
        N_TRIALS,
        exp_params.random_state,
        best_model_save_path=BEST_MODEL_PATH
    )

    logger.info(f"BEST TRIAL F1: {study.best_value:.4f}")
    logger.info(f"Params: {study.best_params}")
    logger.info(f"Best model saved to: {BEST_MODEL_PATH}")