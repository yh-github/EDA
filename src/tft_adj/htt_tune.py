import argparse
import logging
import os
from pathlib import Path

import optuna

from common.config import EmbModel
from common.exp_utils import set_global_seed
from common.log_utils import setup_logging
from htt import HybridTransactionTransformer
from multi.binary.binary_trainer import BinaryMultiTrainer
from multi.config import MultiExpConfig
from multi.tune_multi2 import TuningManager, get_next_study_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hybrid_tune")


class HybridTuningManager(TuningManager):
    def objective(self, trial):
        if self.killer.kill_now:
            raise optuna.TrialPruned()

        # --- Hybrid Specific Search Space ---
        lr = trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.3)

        # The Hybrid model is deeper/heavier, so we might want fewer layers
        num_layers = trial.suggest_int("num_layers", 2, 4)

        # Key param: How much does the model rely on contrastive signal?
        contrastive_weight = trial.suggest_float("contrastive_loss_weight", 0.1, 0.5)

        config = MultiExpConfig(
            learning_rate=lr,
            dropout=dropout,
            num_layers=num_layers,
            contrastive_loss_weight=contrastive_weight,

            # Defaults
            emb_model=EmbModel.MPNET,
            use_cached_embeddings=True,  # Always use cache for speed
            hidden_dim=256,
            num_heads=4,

            # Passthroughs
            num_epochs=self.args.epochs,
            batch_size=self.args.batch_size,
            data_path=self.args.data_path,
            output_dir=self.args.output_dir,
            metric_to_track="cycle_f1",
            task_type="binary",
            use_counter_party=self.data_determined_use_cp
        )

        set_global_seed(config.random_state)

        # Load Cached Datasets
        train_ds, val_ds = self._get_cached_datasets()
        from multi.data import get_dataloader
        train_loader = get_dataloader(self.train_df, config, shuffle=True, dataset=train_ds)
        val_loader = get_dataloader(self.val_df, config, shuffle=False, dataset=val_ds)

        # --- Init Hybrid Model ---
        model = HybridTransactionTransformer(config)

        # Use Standard Binary Trainer
        trainer = BinaryMultiTrainer(model, config)

        trial_save_path = os.path.join(self.args.output_dir, f"temp_trial_{trial.number}.pth")

        best_val_score = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.num_epochs,
            trial=trial,
            stop_callback=lambda: self.killer.kill_now,
            save_path=trial_save_path,
            metric_to_track=config.metric_to_track
        )

        if best_val_score > self.best_global_score:
            logger.info(f"🏆 New Global Best: {best_val_score:.4f}")
            self.best_global_score = best_val_score
            import shutil
            if os.path.exists(trial_save_path):
                shutil.copy(trial_save_path, self.best_global_path)

        if os.path.exists(trial_save_path):
            os.remove(trial_save_path)

        return best_val_score


def main():
    defaults = MultiExpConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=defaults.data_path)
    parser.add_argument("--output_dir", type=str, default="checkpoints/htt")
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--random_state", type=int, default=defaults.random_state)
    parser.add_argument("--downsample", type=float, default=defaults.downsample)
    parser.add_argument("--force_recreate", action="store_true")

    # Hidden args for compatibility
    parser.add_argument("--unfreeze_last_n_layers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--metric_to_track", type=str, default="cycle_f1")
    parser.add_argument("--task_type", type=str, default="binary")

    args = parser.parse_args()

    storage_url = f"sqlite:///{args.output_dir}/tuning.db"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.study_name is None:
        study_name = get_next_study_name(storage_url, base_name="hybrid_tune")
    else:
        study_name = args.study_name

    setup_logging(log_dir=Path('logs/hybrid/'), file_prefix=study_name)
    logger.info(f"Starting Hybrid Tuning: {study_name}")

    manager = HybridTuningManager(args, study_name)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
    )

    study.optimize(manager.objective, n_trials=args.n_trials)


if __name__ == "__main__":
    main()