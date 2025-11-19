import logging
from pathlib import Path

import lightning.pytorch as pl
import optuna
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import CrossEntropy

from config import FieldConfig
from data import create_train_val_test_split
from log_utils import setup_logging
from tft_data import build_tft_dataset, prepare_tft_data

setup_logging(Path("logs/"), "tft_tuning")
logger = logging.getLogger("tft_tuner")

MAX_EPOCHS = 30
N_TRIALS = 50
STUDY_NAME = "tft_optimization_v1"
STORAGE_PATH = f"sqlite:///cache/{STUDY_NAME}.db"


# --- FIX 1: Add arguments to the function signature ---
def objective(trial, train_ds, train_loader, val_loader):
    # 1. Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    attention_head_size = trial.suggest_categorical("attention_head_size", [2, 4])
    hidden_continuous_size = trial.suggest_categorical("hidden_continuous_size", [16, 32, 64])
    gradient_clip_val = trial.suggest_float("gradient_clip_val", 0.01, 1.0)

    # 2. Model (Uses passed 'train_ds')
    tft = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        output_size=2,
        loss=CrossEntropy(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # 3. Trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        gradient_clip_val=gradient_clip_val,
        callbacks=[early_stop_callback, lr_logger],
        enable_progress_bar=False,
        logger=False
    )

    # 4. Fit (Uses passed loaders)
    trainer.fit(
        model=tft,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    val_loss = trainer.callback_metrics.get("val_loss")
    return val_loss.item() if val_loss else float("inf")


if __name__ == "__main__":
    # ... (Loading Code is the same as before) ...
    logger.info("Loading data...")
    field_config = FieldConfig()
    full_df = pd.read_csv("data/rec_data2.csv").dropna(
        subset=[field_config.date, field_config.amount, field_config.text])

    train_df, val_df, _ = create_train_val_test_split(test_size=0.2, val_size=0.2, full_df=full_df, random_state=42)

    train_df_prepped = prepare_tft_data(train_df, field_config)
    val_df_prepped = prepare_tft_data(val_df, field_config)

    train_ds = build_tft_dataset(train_df_prepped, field_config)
    val_ds = TimeSeriesDataSet.from_dataset(train_ds, val_df_prepped, predict=True, stop_randomization=True)

    BATCH_SIZE = 128
    train_loader = train_ds.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=4)
    val_loader = val_ds.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=4)

    logger.info(f"Starting Optuna Study: {STUDY_NAME}")
    study = optuna.create_study(study_name=STUDY_NAME, storage=STORAGE_PATH, load_if_exists=True, direction="minimize")

    # --- FIX 2: Use lambda to inject data into the objective ---
    study.optimize(
        lambda trial: objective(trial, train_ds, train_loader, val_loader),
        n_trials=N_TRIALS
    )

    logger.info("TUNING COMPLETE")
    logger.info(f"Best Params: {study.best_params}")