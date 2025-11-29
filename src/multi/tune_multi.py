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

        val_loader = get_dataloader(self.val_df, config, shuffle=False)


def analyze_mistakes(model, val_df, config, num_examples=5):
    """
    Runs inference on Validation DF and logs the top False Positives and False Negatives.
    """
    logger.info("Running Mistake Analysis on Validation Set...")
    model.eval()

    # We use a non-shuffled loader to keep alignment
    val_loader = get_dataloader(val_df, config, shuffle=False, n_workers=0)
    device = config.device

    fps = []  # False Positives (Pred=1, True=0)
    fns = []  # False Negatives (Pred=0, True=1)

    fc = MultiFieldConfig()

    with torch.no_grad():
        for batch in val_loader:
            batch_gpu = {k: v.to(device) for k, v in batch.items()}

            # Forward
            adj_logits, _, _ = model(batch_gpu)
            probs = torch.sigmoid(adj_logits)

            # CPU conversion
            probs_np = probs.cpu().numpy()
            targets_np = batch['adjacency_target'].numpy()
            indices_np = batch['original_index'].numpy()
            mask_np = batch['padding_mask'].numpy()

            batch_size = probs_np.shape[0]

            for b in range(batch_size):
                # Valid sequence length
                seq_len = int(mask_np[b].sum())

                # Get the sub-matrices for valid items
                p_mat = probs_np[b, :seq_len, :seq_len]
                t_mat = targets_np[b, :seq_len, :seq_len]
                idxs = indices_np[b, :seq_len]

                # Iterate all pairs (upper triangle to avoid dupes/self)
                for i in range(seq_len):
                    for j in range(i + 1, seq_len):
                        prob = float(p_mat[i, j])
                        truth = int(t_mat[i, j])
                        pred = 1 if prob > 0.5 else 0

                        # Identify Mistakes
                        if pred == 1 and truth == 0:
                            # False Positive: Model thinks i and j are same pattern
                            row_i = val_df.iloc[idxs[i]]
                            row_j = val_df.iloc[idxs[j]]
                            fps.append({
                                'prob': prob,
                                'txt1': row_i[fc.text], 'amt1': row_i[fc.amount],
                                'txt2': row_j[fc.text], 'amt2': row_j[fc.amount]
                            })

                        elif pred == 0 and truth == 1:
                            # False Negative: Model missed the connection
                            row_i = val_df.iloc[idxs[i]]
                            row_j = val_df.iloc[idxs[j]]
                            fns.append({
                                'prob': prob,
                                'txt1': row_i[fc.text], 'amt1': row_i[fc.amount],
                                'txt2': row_j[fc.text], 'amt2': row_j[fc.amount]
                            })

    # Sort and Log
    logger.info("-" * 50)
    logger.info(f"MISTAKE REPORT (Found {len(fps)} FPs, {len(fns)} FNs)")

    # 1. Top False Positives (High Confidence, Wrong)
    fps_sorted = sorted(fps, key=lambda z: z['prob'], reverse=True)[:num_examples]
    if fps_sorted:
        logger.info("\n>>> Top False Positives (Model thought these were same):")
        for x in fps_sorted:
            logger.info(f"  [{x['prob']:.4f}] '{x['txt1']}' (${x['amt1']}) <--> '{x['txt2']}' (${x['amt2']})")

    # 2. Top False Negatives (Low Confidence, Wrong)
    # We sort by prob ascending (closest to 0)
    fns_sorted = sorted(fns, key=lambda z: z['prob'])[:num_examples]
    if fns_sorted:
        logger.info("\n>>> Top False Negatives (Model missed these):")
        for x in fns_sorted:
            logger.info(f"  [{x['prob']:.4f}] '{x['txt1']}' (${x['amt1']}) <--> '{x['txt2']}' (${x['amt2']})")

    logger.info("-" * 50)


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
    parser.add_argument("--epochs", type=int, default=defaults.num_epochs)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--random_state", type=int, default=defaults.random_state)
    parser.add_argument("--downsample", type=float, default=defaults.downsample)
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
    logger.info(f"Best Trial Score: {study.best_value}")
    logger.info("Best Params:")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")

    # --- ANALYSIS 1: Hyperparameters ---
    analyze_study_importance(study)

    # --- ANALYSIS 2: Best Model & Mistakes ---
    logger.info("=" * 60)
    logger.info("STARTING FINAL MODEL TRAINING (Train + Val -> Test)")
    logger.info("=" * 60)

    # 1. Merge Train and Val
    full_train_df = pd.concat([manager.train_df, manager.val_df])
    logger.info(
        f"Combined Train ({len(manager.train_df)}) + Val ({len(manager.val_df)}) = {len(full_train_df)} samples")

    # 2. Reconstruct Config from Best Params + Args
    best_params = study.best_params

    # FIX: Handle potential missing keys if tuning space changes, or mapping keys
    cw_param = best_params.get("contrastive_weight") or best_params.get("contrastive_loss_weight") or 0.1

    defaults = MultiExpConfig()
    final_config = MultiExpConfig(
        # Tuned Params
        learning_rate=best_params["learning_rate"],
        dropout=best_params["dropout"],
        num_layers=best_params["num_layers"],
        num_heads=best_params["num_heads"],
        hidden_dim=best_params["hidden_dim"],

        # Mapped Correctly
        contrastive_loss_weight=cw_param,

        # Fixed Params
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        data_path=args.data_path,
        output_dir=args.output_dir,
        early_stopping_patience=defaults.early_stopping_patience,

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
    if not manager.killer.kill_now:
        save_path = os.path.join(args.output_dir, f"best_model_{study_name}.pth")

        final_trainer.fit(
            train_loader=final_train_loader,
            val_loader=test_loader,
            epochs=final_config.num_epochs,
            save_path=save_path,
            stop_callback=lambda: manager.killer.kill_now,
            metric_to_track='pr_auc'
        )

        # 6. Final Metrics
        logger.info("-" * 60)
        logger.info("FINAL TEST SET RESULTS")
        logger.info("-" * 60)
        final_metrics = final_trainer.evaluate(test_loader)
        logger.info(f"Precision: {final_metrics['precision']:.4f}")
        logger.info(f"Recall:    {final_metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {final_metrics['f1']:.4f}")
        logger.info(f"PR-AUC:    {final_metrics['pr_auc']:.4f}")
        logger.info(f"ROC-AUC:   {final_metrics['roc_auc']:.4f}")
        logger.info(f"Loss:      {final_metrics['val_loss']:.4f}")

        logger.info(f"Final model saved to {save_path}")

        # 7. Analyze Mistakes (On Test Set now)
        analyze_mistakes(final_model, manager.test_df, final_config)


if __name__ == "__main__":
    main()