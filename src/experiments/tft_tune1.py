import logging
from pathlib import Path
import optuna
import pandas as pd
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import CrossEntropy
from pytorch_lightning.callbacks import EarlyStopping

# --- Imports from your new files ---
from config import FieldConfig, ExperimentConfig
from data import create_train_val_test_split  # Reuse your existing split logic
# Setup Logging
from log_utils import setup_logging
from tft_data import build_tft_dataset, prepare_tft_data

setup_logging(Path("logs/"), "tft_tuning")
logger = logging.getLogger("tft_tuner")

# Constants
MAX_EPOCHS = 30
N_TRIALS = 50  # How many experiments to run overnight
STUDY_NAME = "tft_optimization_v1"
STORAGE_PATH = f"sqlite:///cache/{STUDY_NAME}.db"  # Persistent storage


def objective(trial):
    # 1. Suggest Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    attention_head_size = trial.suggest_categorical("attention_head_size", [2, 4])
    hidden_continuous_size = trial.suggest_categorical("hidden_continuous_size", [16, 32, 64])

    # Gradient Clipping is crucial for LSTMs/Transformers
    gradient_clip_val = trial.suggest_float("gradient_clip_val", 0.01, 1.0)

    # 2. Data Loading (Cached/Global to save time if possible, but safe to rebuild)
    # Note: We rely on global `train_ds` and `val_dataloader` to avoid reloading DF every trial

    # 3. Initialize Model
    tft = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        output_size=2,  # Binary Classification
        loss=CrossEntropy(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # 4. Trainer with Pruning
    # PyTorch Lightning integration with Optuna for early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=4, verbose=False, mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        gradient_clip_val=gradient_clip_val,
        callbacks=[early_stop_callback],
        enable_progress_bar=False,  # Reduce log noise
        logger=False  # Don't create version folders for every trial
    )

    # 5. Fit
    trainer.fit(
        tft,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # 6. Return Metric
    val_loss = trainer.callback_metrics["val_loss"].item()
    return val_loss


if __name__ == "__main__":
    # --- 1. Load & Split Data ONCE ---
    logger.info("Loading data...")
    field_config = FieldConfig()
    full_df = pd.read_csv("data/rec_data2.csv")

    # Filter/Clean
    full_df = full_df.dropna(subset=[field_config.date, field_config.amount, field_config.text])

    # Split using your existing logic (Account-aware)
    train_df, val_df, _ = create_train_val_test_split(
        test_size=0.2, val_size=0.2, full_df=full_df, random_state=ExperimentConfig().random_state
    )

    # --- Prepare BOTH DataFrames ---
    logger.info("Preparing data for TFT (sorting, time_idx)...")
    train_df_prepped = prepare_tft_data(train_df, field_config)
    val_df_prepped = prepare_tft_data(val_df, field_config)

    # 2. Build Dataset Definition (from Training data)
    train_ds = build_tft_dataset(train_df_prepped, field_config)

    # 3. Create Validation Dataset (using definition from Train)
    # Now val_df_prepped has 'time_idx', so this will work!
    val_ds = TimeSeriesDataSet.from_dataset(
        train_ds,
        val_df_prepped,
        predict=True,
        stop_randomization=True
    )

    # Create Loaders
    BATCH_SIZE = 128
    train_loader = train_ds.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=4)
    val_loader = val_ds.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=4)

    # --- 3. Run Optuna ---
    logger.info(f"Starting Optuna Study: {STUDY_NAME}")
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_PATH,
        load_if_exists=True,
        direction="minimize"
    )

    study.optimize(objective, n_trials=N_TRIALS)

    # --- 4. Report & Save Best ---
    logger.info("=" * 50)
    logger.info("TUNING COMPLETE")
    logger.info(f"Best Value (Val Loss): {study.best_value}")
    logger.info(f"Best Params: {study.best_params}")
    logger.info("=" * 50)

    # Optional: Retrain best model fully or save the checkpoint logic