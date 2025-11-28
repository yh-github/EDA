import argparse
import logging
import os
import pickle
import signal
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from common.data import create_train_val_test_split, clean_text
from common.exp_utils import set_global_seed
from multi.config import MultiExpConfig, MultiFieldConfig
from multi.data import get_dataloader
from multi.encoder import TransactionTransformer
from multi.trainer import MultiTrainer

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hyper_tune")


class GracefulKiller:
    """
    Catches SIGINT (Ctrl+C) and SIGTERM to allow the current trial to finish
    or the study to save its state before exiting.
    """

    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        logger.warning(f"Received signal {signum}. Finishing current step/trial and exiting...")
        self.kill_now = True


class TuningManager:
    def __init__(self, args):
        self.args = args
        self.killer = GracefulKiller()

        # Determine Cache Path
        self.cache_dir = Path("cache/data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        cache_key = f"{args.random_state}__{args.downsample}"
        self.cache_path = self.cache_dir / f"split_{cache_key}.pkl"

        # Load or Create Data
        self.train_df, self.val_df, self.test_df = self._load_or_create_data()

    def _load_or_create_data(self):
        if self.cache_path.exists() and not self.args.force_recreate:
            logger.info(f"Loading cached data splits from {self.cache_path}...")
            with open(self.cache_path, 'rb') as f:
                data = pickle.load(f)
            return data['train'], data['val'], data['test']

        logger.info("Cache miss or force recreate. Loading raw data...")
        if not os.path.exists(self.args.data_path):
            raise FileNotFoundError(f"Data file not found: {self.args.data_path}")

        df = pd.read_csv(self.args.data_path, low_memory=False)
        field_config = MultiFieldConfig()

        # Basic Preprocessing
        df[field_config.accountId] = df[field_config.accountId].astype(str)
        df[field_config.trId] = df[field_config.trId].astype(str)
        df[field_config.text] = clean_text(df[field_config.text])

        # Downsample
        if 0.0 < self.args.downsample < 1.0:
            logger.info(f"Downsampling to {self.args.downsample:.0%} of accounts...")
            account_ids = df[field_config.accountId].unique()
            rng = np.random.default_rng(self.args.random_state)
            n_select = max(1, int(len(account_ids) * self.args.downsample))
            selected_ids = rng.choice(account_ids, size=n_select, replace=False)
            df = df[df[field_config.accountId].isin(selected_ids)].copy()

        # Split
        logger.info("Splitting data...")
        train_df, val_df, test_df = create_train_val_test_split(
            test_size=0.1,
            val_size=0.1,
            full_df=df,
            random_state=self.args.random_state,
            field_config=field_config
        )

        # Save
        logger.info(f"Saving splits to {self.cache_path}...")
        with open(self.cache_path, 'wb') as f:
            pickle.dump({'train': train_df, 'val': val_df, 'test': test_df}, f)

        return train_df, val_df, test_df

    def objective(self, trial):
        # 1. Check for Stop Signal at start of trial
        if self.killer.kill_now:
            logger.warning("Stop signal received. Stopping study.")
            trial.study.stop()
            raise optuna.TrialPruned()

        # 2. Sample Hyperparameters
        lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        num_layers = trial.suggest_int("num_layers", 1, 4)
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])

        # Loss params
        contrastive_weight = trial.suggest_float("contrastive_weight", 0.0, 0.5)

        # 3. Configure Experiment
        config = MultiExpConfig(
            learning_rate=lr,
            dropout=dropout,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            contrastive_loss_weight=contrastive_weight,

            # Constants from Args
            num_epochs=self.args.epochs,
            batch_size=self.args.batch_size,
            data_path=self.args.data_path,
            output_dir=self.args.output_dir,
            use_counter_party=True  # Can be tuned too if needed
        )

        set_global_seed(config.random_state)

        # 4. Data Loaders (Using cached DF)
        train_loader = get_dataloader(self.train_df, config, shuffle=True)
        val_loader = get_dataloader(self.val_df, config, shuffle=False)

        # 5. Model & Trainer
        model = TransactionTransformer(config)

        # Calculate class weights if needed (simple heuristic for now)
        pos_weight = 2.5  # Could calculate dynamically from train_df

        trainer = MultiTrainer(model, config, pos_weight=pos_weight)

        best_val_f1 = 0.0

        # 6. Training Loop
        for epoch in range(config.num_epochs):
            # Check signal
            if self.killer.kill_now:
                trainer.request_stop()

            train_loss = trainer.train_epoch(train_loader, epoch + 1)
            metrics = trainer.evaluate(val_loader)
            val_f1 = metrics['f1']

            # Report to Optuna
            trial.report(val_f1, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                # Optional: Save best model for this trial

            # Stop early if requested
            if trainer.stop_requested:
                logger.info(f"Trial stopped early at epoch {epoch + 1}. Returning best F1 so far.")
                break

        return best_val_f1


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Hyperparameter Tuning")
    parser.add_argument("--data_path", type=str, default="data/all_data.csv")
    parser.add_argument("--output_dir", type=str, default="checkpoints/tuning")
    parser.add_argument("--study_name", type=str, default="multi_opt_v1")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--random_state", type=int, default=112025)
    parser.add_argument("--downsample", type=float, default=0.3)
    parser.add_argument("--force_recreate", action="store_true", help="Force recreate data splits")
    args = parser.parse_args()

    manager = TuningManager(args)

    # Optuna Storage (SQLite for persistence)
    storage_url = f"sqlite:///{args.output_dir}/tuning.db"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
    )

    logger.info(f"Starting tuning study '{args.study_name}' with {args.n_trials} trials.")

    try:
        study.optimize(manager.objective, n_trials=args.n_trials)
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt caught in main loop. Saving study state...")

    logger.info("Tuning Complete.")
    logger.info(f"Best Trial F1: {study.best_value}")
    logger.info("Best Params:")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")

    # Optionally: Train final model on Train+Val and test on Test set here
    # using manager.test_df and study.best_params


if __name__ == "__main__":
    main()