import os
from common.config import EmbModel
from common.log_utils import setup_logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import logging
import signal
import pickle
import re
import shutil
from pathlib import Path
import optuna
import torch
from multi.config import MultiExpConfig, MultiFieldConfig
from multi.encoder import TransactionTransformer
from multi.trainer import MultiTrainer
from multi.data import get_dataloader
from multi.data_utils import load_and_prepare_data
from common.data import create_train_val_test_split
from common.exp_utils import set_global_seed, get_git_info

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hyper_tune")


# TODO create experiment files with only the 'suggested' parameters
# TODO  n_startup_trials

class GracefulKiller:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        logger.warning(f"Received signal {signum}. Finishing current step/trial and exiting...")
        self.kill_now = True

DATA_CACHE_BASE = Path("cache/data")

def get_data_cache_path(random_state:int, downsample:float, base:Path = DATA_CACHE_BASE) -> Path:
    cache_key = f"{random_state}__{downsample}"
    return base/f"split_{cache_key}.pkl"


class TuningManager:

    def __init__(self, args, study_name):
        self.args = args
        self.study_name = study_name
        self.killer = GracefulKiller()

        self.best_global_score = -1.0
        self.best_global_path = Path(args.output_dir) / f"best_model_{study_name}.pth"

        # Determine Cache Path
        DATA_CACHE_BASE.mkdir(parents=True, exist_ok=True)

        self.cache_path = get_data_cache_path(
            random_state=args.random_state, downsample=args.downsample
        )

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

        # --- Sample Hyperparameters ---

        # 1. Unfreeze Layers: 0 (Frozen), 1 (Top), 2 (Top 2)
        unfreeze = trial.suggest_categorical("unfreeze_last_n_layers", [1, 2])

        # 2. Conditional Learning Rate Strategy
        if unfreeze > 0:
            lr_min = 1e-5
            lr_max = 1e-3
        else:
            lr_min = 1e-4
            lr_max = 1e-3

        lr = trial.suggest_float("learning_rate", lr_min, lr_max, log=True)

        # 3. Normalization Strategy
        norm_type = trial.suggest_categorical("normalization_type", ["layer_norm", "rms_norm"])

        # 4. Standard architecture params
        dropout = trial.suggest_float("dropout", 0.3, 0.45)
        num_layers = trial.suggest_int("num_layers", 5, 8)
        num_heads = trial.suggest_categorical("num_heads", [4, 8])
        hidden_dim = trial.suggest_categorical("hidden_dim", [256])
        contrastive_loss_weight = trial.suggest_float("contrastive_loss_weight", 0.4, 0.5)

        defaults = MultiExpConfig()
        # noinspection PyTypeChecker,PyTypeHints
        emb_str: str = EmbModel[self.args.text_emb].value

        config = MultiExpConfig(
            learning_rate=lr,
            dropout=dropout,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            contrastive_loss_weight=contrastive_loss_weight,

            # Merged Params
            unfreeze_last_n_layers=unfreeze,
            normalization_type=norm_type,

            num_epochs=self.args.epochs,
            batch_size=self.args.batch_size,
            data_path=self.args.data_path,
            output_dir=self.args.output_dir,
            text_encoder_model=emb_str,
            metric_to_track=self.args.metric_to_track,
            early_stopping_patience=defaults.early_stopping_patience,
            use_counter_party=self.data_determined_use_cp
        )

        set_global_seed(config.random_state)

        train_loader = get_dataloader(self.train_df, config, shuffle=True)
        val_loader = get_dataloader(self.val_df, config, shuffle=False)

        model = TransactionTransformer(config)
        trainer = MultiTrainer(model, config)

        # Temporary path for this specific trial
        trial_save_path = os.path.join(self.args.output_dir, f"temp_trial_{trial.number}.pth")

        # Run unified training loop
        # We pass the metric to track here
        best_val_score = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.num_epochs,
            trial=trial,
            stop_callback=lambda: self.killer.kill_now,
            save_path=trial_save_path,
            metric_to_track=config.metric_to_track
        )

        # Global Best Model Saving Logic
        if best_val_score > self.best_global_score:
            logger.info(f"🏆 New Global Best Score ({self.args.metric_to_track}): {best_val_score:.4f} (Trial {trial.number})")
            self.best_global_score = best_val_score
            # Promote temp file to global best file
            if os.path.exists(trial_save_path):
                shutil.copy(trial_save_path, self.best_global_path)
                logger.info(f"Saved global best model to {self.best_global_path}")

        # Cleanup temp file
        if os.path.exists(trial_save_path):
            os.remove(trial_save_path)

        if trainer.stop_requested:
            logger.info("Trial stopped by user signal.")

        return best_val_score


def analyze_study_importance(study):
    try:
        import optuna.importance
        logger.info("\n=== Hyperparameter Importance ===")
        importance = optuna.importance.get_param_importances(study)
        for param, score in importance.items():
            logger.info(f"  {param:<25}: {score:.4f}")
    except Exception as e:
        logger.warning(f"Could not calculate parameter importance: {e}")


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
    defaults = MultiExpConfig()
    parser = argparse.ArgumentParser(description="Multi-Model Hyperparameter Tuning")
    parser.add_argument("--data_path", type=str, default=defaults.data_path)
    parser.add_argument("--output_dir", type=str, default=defaults.output_dir)
    parser.add_argument("--study_name", type=str, default=None, help="If None, auto-increments multi_tune_{i}")
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--unfreeze_last_n_layers", type=int, default=defaults.unfreeze_last_n_layers)
    parser.add_argument("--epochs", type=int, default=defaults.num_epochs)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--random_state", type=int, default=defaults.random_state)
    parser.add_argument("--downsample", type=float, default=defaults.downsample)
    parser.add_argument("--learning_rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--force_recreate", action="store_true", help="Force recreate data splits")
    parser.add_argument("--text_emb", type=str, default="MPNET", # TODO fix in config
                        help=f"Model name. Accepts EmbModel keys (e.g., 'MPNET', 'ALBERT').")

    parser.add_argument("--metric_to_track", type=str, default="pr_auc", choices=["pr_auc", "cycle_f1"],
                        help="Metric to optimize: 'pr_auc' (Adjacency/Clustering) or 'cycle_f1' (Detection)")

    args = parser.parse_args()

    # Optuna Storage (SQLite for persistence)
    storage_url = f"sqlite:///{args.output_dir}/tuning.db"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Auto-naming logic
    if args.study_name is None:
        study_name = get_next_study_name(storage_url)
    else:
        study_name = args.study_name

    setup_logging(log_dir=Path('logs/multi/'), file_prefix=study_name)

    logger.info(f"Using Study Name: {study_name} {get_git_info()} PID={os.getpid()}")
    logger.info(f"Optimization Target: {args.metric_to_track}")

    manager = TuningManager(args, study_name)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
    )

    # Save Process Info to Study User Attributes (Overwrite to update last run info)
    current_pid = os.getpid()

    # Maintain history of PIDs
    all_pids = study.user_attrs.get("all_pids", [])
    if not isinstance(all_pids, list):
        all_pids = []

    if current_pid not in all_pids:
        all_pids.append(current_pid)
        study.set_user_attr("all_pids", all_pids)

    logger.info(f"Starting tuning study '{study_name}' with {args.n_trials} trials.")

    study.set_user_attr("git_info", get_git_info())
    study.set_user_attr("target_metric", args.metric_to_track)

    try:
        study.optimize(manager.objective, n_trials=args.n_trials)
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt caught in main loop. Saving study state...")

    logger.info("Tuning Complete.")
    if len(study.trials) > 0:
        logger.info(f"Best Trial Score ({args.metric_to_track}): {study.best_value}")
        logger.info("Best Params:")
        for k, v in study.best_params.items():
            logger.info(f"  {k}: {v}")

        # --- ANALYSIS 1: Hyperparameters ---
        analyze_study_importance(study)

        # --- ANALYSIS 2: Re-Load Best Model for Detailed Analysis ---
        # logger.info("=" * 60)
        # logger.info("LOADING BEST MODEL FOR DETAILED ANALYSIS")
        # logger.info("=" * 60)
        #
        # if manager.best_global_path.exists():
        #     # Create a basic config just to initialize the predictor/analysis tools
        #     # The saved checkpoint contains the exact config
        #     checkpoint = torch.load(manager.best_global_path, map_location=torch.device('cpu'), weights_only=False)
        #     loaded_config = checkpoint['config']
        #
        #     model = TransactionTransformer(loaded_config)
        #     model.load_state_dict(checkpoint['state_dict'])
        #     model.to(get_device())
        #
        #     # Analyze on Test Set
        #     logger.info("Initializing Test Loader for Analysis...")
        #     test_loader = get_dataloader(manager.test_df, loaded_config, shuffle=False, n_workers=0)
        #     analyze_adjacency_mistakes(model, test_loader, loaded_config)
        #     analyze_classification_mistakes(model, test_loader, loaded_config)
        # else:
        #     logger.warning("Best model file not found. Skipping analysis.")

    else:
        logger.warning("No trials completed.")


if __name__ == "__main__":
    main()