import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import logging
import signal
import pickle
import re
from pathlib import Path
import optuna
import pandas as pd
import torch
from multi.config import MultiExpConfig, MultiFieldConfig
from multi.encoder import TransactionTransformer
from multi.trainer import MultiTrainer
from multi.data import get_dataloader
from multi.data_utils import load_and_prepare_data
from common.data import create_train_val_test_split
from common.exp_utils import set_global_seed

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hyper_tune")


class GracefulKiller:
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

        # IMPORTANT: We need to know if data processing disabled CP
        self.data_determined_use_cp = True

        # Load or Create Data
        self.train_df, self.val_df, self.test_df = self._load_or_create_data()

    def _load_or_create_data(self):
        if self.cache_path.exists() and not self.args.force_recreate:
            logger.info(f"Loading cached data splits from {self.cache_path}...")
            with open(self.cache_path, 'rb') as f:
                data = pickle.load(f)

            # Restore the CP decision from the cache
            self.data_determined_use_cp = data.get('use_counter_party', True)
            logger.info(f"Cached data indicates use_counter_party={self.data_determined_use_cp}")

            return data['train'], data['val'], data['test']

        logger.info("Cache miss or force recreate. Running data pipeline...")

        # 1. Create a temp config to pass args to the loader
        # We assume use_counter_party is True initially, the loader might flip it
        temp_config = MultiExpConfig(
            data_path=self.args.data_path,
            downsample=self.args.downsample,
            random_state=self.args.random_state,
            use_counter_party=True
        )

        # 2. Use shared utility to load, clean, downsample, and check features
        df = load_and_prepare_data(temp_config)

        # 3. Capture the CP decision made by the loader
        self.data_determined_use_cp = temp_config.use_counter_party

        # 4. Split
        logger.info("Splitting data...")
        train_df, val_df, test_df = create_train_val_test_split(
            test_size=0.1,
            val_size=0.1,
            full_df=df,
            random_state=self.args.random_state,
            field_config=MultiFieldConfig()
        )

        # 5. Save with metadata
        logger.info(f"Saving splits to {self.cache_path}...")
        with open(self.cache_path, 'wb') as f:
            pickle.dump({
                'train': train_df,
                'val': val_df,
                'test': test_df,
                'use_counter_party': self.data_determined_use_cp
            }, f)

        return train_df, val_df, test_df

    def objective(self, trial):
        if self.killer.kill_now:
            logger.warning("Stop signal received. Stopping study.")
            trial.study.stop()
            raise optuna.TrialPruned()

        # Sample Hyperparameters
        lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        num_layers = trial.suggest_int("num_layers", 1, 4)
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
        contrastive_weight = trial.suggest_float("contrastive_weight", 0.0, 0.5)

        config = MultiExpConfig(
            learning_rate=lr,
            dropout=dropout,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            contrastive_loss_weight=contrastive_weight,

            num_epochs=self.args.epochs,
            batch_size=self.args.batch_size,
            data_path=self.args.data_path,
            output_dir=self.args.output_dir,

            # CRITICAL: Use the decision made during data loading
            use_counter_party=self.data_determined_use_cp
        )

        set_global_seed(config.random_state)

        train_loader = get_dataloader(self.train_df, config, shuffle=True)
        val_loader = get_dataloader(self.val_df, config, shuffle=False)

        model = TransactionTransformer(config)
        pos_weight = 2.5

        trainer = MultiTrainer(model, config, pos_weight=pos_weight)

        best_val_f1 = 0.0

        for epoch in range(config.num_epochs):
            if self.killer.kill_now:
                trainer.request_stop()

            _ = trainer.train_epoch(train_loader, epoch + 1)
            metrics = trainer.evaluate(val_loader)
            val_f1 = metrics['f1']

            trial.report(val_f1, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1

            if trainer.stop_requested:
                logger.info(f"Trial stopped early at epoch {epoch + 1}. Returning best F1 so far.")
                break

        return best_val_f1


def get_next_study_name(storage_url: str, base_name: str = "multi_tune") -> str:
    try:
        summaries = optuna.study.get_all_study_summaries(storage=storage_url)
        existing_names = [s.study_name for s in summaries]
    except Exception:
        existing_names = []

    pattern = re.compile(rf"^{base_name}_(\d+)$")

    max_idx = 0
    for name in existing_names:
        match = pattern.match(name)
        if match:
            idx = int(match.group(1))
            if idx > max_idx:
                max_idx = idx

    next_idx = max_idx + 1
    return f"{base_name}_{next_idx}"


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Hyperparameter Tuning")
    parser.add_argument("--data_path", type=str, default="data/all_data.csv")
    parser.add_argument("--output_dir", type=str, default="checkpoints/tuning")
    parser.add_argument("--study_name", type=str, default=None, help="If None, auto-increments multi_tune_{i}")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--random_state", type=int, default=112025)
    parser.add_argument("--downsample", type=float, default=0.5)
    parser.add_argument("--force_recreate", action="store_true", help="Force recreate data splits")
    args = parser.parse_args()

    manager = TuningManager(args)

    # Optuna Storage (SQLite for persistence)
    storage_url = f"sqlite:///{args.output_dir}/tuning.db"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Auto-naming logic
    if args.study_name is None:
        study_name = get_next_study_name(storage_url)
    else:
        study_name = args.study_name

    logger.info(f"Using Study Name: {study_name}")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
    )

    logger.info(f"Starting tuning study '{study_name}' with {args.n_trials} trials.")

    try:
        study.optimize(manager.objective, n_trials=args.n_trials)
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt caught in main loop. Saving study state...")

    logger.info("Tuning Complete.")
    logger.info(f"Best Trial F1: {study.best_value}")
    logger.info("Best Params:")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")

    # --- FINAL EVALUATION ---
    logger.info("=" * 60)
    logger.info("STARTING FINAL MODEL TRAINING (Train + Val -> Test)")
    logger.info("=" * 60)

    # 1. Merge Train and Val
    full_train_df = pd.concat([manager.train_df, manager.val_df])
    logger.info(
        f"Combined Train ({len(manager.train_df)}) + Val ({len(manager.val_df)}) = {len(full_train_df)} samples")

    # 2. Reconstruct Config from Best Params + Args
    best_params = study.best_params
    final_config = MultiExpConfig(
        # Tuned Params
        learning_rate=best_params["learning_rate"],
        dropout=best_params["dropout"],
        num_layers=best_params["num_layers"],
        num_heads=best_params["num_heads"],
        hidden_dim=best_params["hidden_dim"],
        contrastive_loss_weight=best_params["contrastive_loss_weight"],

        # Fixed Params
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        data_path=args.data_path,
        output_dir=args.output_dir,

        # Ensure final model respects data capabilities
        use_counter_party=manager.data_determined_use_cp
    )
    set_global_seed(final_config.random_state)

    # 3. Create Loaders
    logger.info("Creating DataLoaders...")
    final_train_loader = get_dataloader(full_train_df, final_config, shuffle=True)
    test_loader = get_dataloader(manager.test_df, final_config, shuffle=False)

    # 4. Initialize Final Model
    logger.info(
        f"Initializing Final Model with dim={final_config.hidden_dim}, heads={final_config.num_heads}, layers={final_config.num_layers}")
    final_model = TransactionTransformer(final_config)
    final_trainer = MultiTrainer(final_model, final_config, pos_weight=2.5)

    # 5. Train
    # We allow the killer to stop this too if user hits Ctrl+C
    for epoch in range(final_config.num_epochs):
        if manager.killer.kill_now:
            logger.warning("Stop signal received during final training. Exiting.")
            break

        train_loss = final_trainer.train_epoch(final_train_loader, epoch + 1)
        metrics = final_trainer.evaluate(test_loader)

        logger.info(f"Final Model Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Test F1: {metrics['f1']:.4f}")

    # 6. Final Metrics
    if not manager.killer.kill_now:
        logger.info("-" * 60)
        logger.info("FINAL TEST SET RESULTS")
        logger.info("-" * 60)
        final_metrics = final_trainer.evaluate(test_loader)
        logger.info(f"Precision: {final_metrics['precision']:.4f}")
        logger.info(f"Recall:    {final_metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {final_metrics['f1']:.4f}")
        logger.info(f"Loss:      {final_metrics['val_loss']:.4f}")

        # Save Model
        # Include study name in filename to avoid overwrites
        save_path = os.path.join(args.output_dir, f"best_model_{study_name}.pth")
        torch.save({
            "config": final_config,
            "state_dict": final_model.state_dict(),
            "best_f1": final_metrics['f1'],
            "params": best_params
        }, save_path)
        logger.info(f"Final model saved to {save_path}")


if __name__ == "__main__":
    main()