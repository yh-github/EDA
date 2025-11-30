import os

from common.config import EmbModel

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


class GracefulKiller:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        logger.warning(f"Received signal {signum}. Finishing current step/trial and exiting...")
        self.kill_now = True


class TuningManager:
    def __init__(self, args, study_name):
        self.args = args
        self.study_name = study_name
        self.killer = GracefulKiller()

        self.best_global_score = -1.0
        self.best_global_path = Path(args.output_dir) / f"best_model_{study_name}.pth"

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

        # --- Sample Hyperparameters ---

        # 1. Learning Rate: User's logic (Dynamic centering)
        lr = trial.suggest_float("learning_rate", self.args.learning_rate / 5, self.args.learning_rate * 5, log=True)

        # 2. Unfreeze Layers: Tune this instead of hardcoding, to find the "Goldilocks" zone
        unfreeze = trial.suggest_categorical("unfreeze_last_n_layers", [0, 1, 2])

        # 3. Normalization Strategy (New)
        norm_type = trial.suggest_categorical("normalization_type", ["layer_norm", "rms_norm"])

        # 4. Standard architecture params
        dropout = trial.suggest_float("dropout", 0.2, 0.4)
        num_layers = trial.suggest_int("num_layers", 3, 6)
        num_heads = trial.suggest_categorical("num_heads", [4, 8])
        hidden_dim = trial.suggest_categorical("hidden_dim", [256])
        contrastive_loss_weight = trial.suggest_float("contrastive_loss_weight", 0.2, 0.5)

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
        best_val_score = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.num_epochs,
            trial=trial,
            stop_callback=lambda: self.killer.kill_now,
            save_path=trial_save_path
        )

        # Global Best Model Saving Logic
        if best_val_score > self.best_global_score:
            logger.info(f"🏆 New Global Best Score: {best_val_score:.4f} (Trial {trial.number})")
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


def analyze_classification_mistakes(model, val_df, config, num_examples=5):
    """
    Analyzes 'isRecurring' classification errors specifically.
    FP: Model says Recurring, Truth says None.
    FN: Model says None, Truth says Recurring.
    """
    logger.info("Running Cycle/Classification Mistake Analysis...")
    model.eval()
    val_loader = get_dataloader(val_df, config, shuffle=False, n_workers=0)
    device = config.device
    fc = MultiFieldConfig()

    fps = []
    fns = []

    with torch.no_grad():
        for batch in val_loader:
            batch_gpu = {k: v.to(device) for k, v in batch.items()}

            # Forward
            _, cycle_logits, _ = model(batch_gpu)

            # Predictions: [Batch, Seq]
            preds = torch.argmax(cycle_logits, dim=-1)
            targets = batch_gpu['cycle_target']
            padding_mask = batch_gpu['padding_mask']

            probs = torch.softmax(cycle_logits, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1)

            # Move to CPU
            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()
            mask_np = padding_mask.cpu().numpy()
            indices_np = batch['original_index'].numpy()
            probs_np = max_probs.cpu().numpy()

            batch_size, seq_len = preds.shape

            for b in range(batch_size):
                for s in range(seq_len):
                    if not mask_np[b, s]: continue

                    pred_cls = preds_np[b, s]
                    true_cls = targets_np[b, s]
                    prob = probs_np[b, s]

                    # Convert to Binary Logic: 0 is None/Noise, >0 is Recurring
                    pred_is_rec = pred_cls > 0
                    true_is_rec = true_cls > 0

                    if pred_is_rec == true_is_rec:
                        continue

                    # Get original row
                    orig_idx = indices_np[b, s]
                    row = val_df.iloc[orig_idx]

                    item = {
                        'txt': row[fc.text],
                        'amt': row[fc.amount],
                        'prob': prob,
                        'pred_cycle': pred_cls,
                        'true_cycle': true_cls
                    }

                    if pred_is_rec and not true_is_rec:
                        fps.append(item)  # False Positive (Spam marked as Recurring)
                    elif not pred_is_rec and true_is_rec:
                        fns.append(item)  # False Negative (Recurring missed)

    # Log Results
    logger.info("-" * 60)
    logger.info(f"CLASSIFICATION MISTAKES (isRecurring): FP={len(fps)}, FN={len(fns)}")

    # Sort by confidence (High prob = worse mistake)
    fps_sorted = sorted(fps, key=lambda x: x['prob'], reverse=True)[:num_examples]
    if fps_sorted:
        logger.info("\n>>> Top False Positives (Predicted Recurring, Actually Noise):")
        for x in fps_sorted:
            logger.info(f"  [{x['prob']:.2f}] Amt:{x['amt']:<8} Desc:'{x['txt']}'")

    # Sort by confidence (High prob of being 0 means it confidently thought it was noise)
    fns_sorted = sorted(fns, key=lambda x: x['prob'], reverse=True)[:num_examples]
    if fns_sorted:
        logger.info("\n>>> Top False Negatives (Predicted Noise, Actually Recurring):")
        for x in fns_sorted:
            logger.info(f"  [{x['prob']:.2f}] Amt:{x['amt']:<8} Desc:'{x['txt']}' (TrueCycle: {x['true_cycle']})")

    logger.info("-" * 60)


def analyze_adjacency_mistakes(model, val_df, config, num_examples=5):
    """
    Runs inference on Validation DF and logs the top False Positives and False Negatives for Adjacency.
    """
    logger.info("Running Adjacency Mistake Analysis...")
    model.eval()

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

            preds = (probs > 0.5).float()
            targets = batch_gpu['adjacency_target']

            # Create masks
            b_size, s_len, _ = probs.shape
            triu_mask = torch.triu(torch.ones(s_len, s_len, device=device), diagonal=1).bool()
            triu_mask = triu_mask.unsqueeze(0).expand(b_size, -1, -1)

            valid_mask = batch_gpu['padding_mask'].unsqueeze(1) & batch_gpu['padding_mask'].unsqueeze(2)
            final_mask = triu_mask & valid_mask

            # Find mismatches: (Pred != Target) & Valid & UpperTri
            mismatches = (preds != targets) & final_mask

            # Get indices of mismatches
            b_indices, i_indices, j_indices = torch.where(mismatches)

            if len(b_indices) == 0:
                continue

            # Move necessary data to CPU for logging
            b_indices = b_indices.cpu().numpy()
            i_indices = i_indices.cpu().numpy()
            j_indices = j_indices.cpu().numpy()

            probs_np = probs.detach().cpu().numpy()
            targets_np = targets.cpu().numpy()
            indices_np = batch['original_index'].numpy()  # CPU already

            for k in range(len(b_indices)):
                b, i, j = b_indices[k], i_indices[k], j_indices[k]

                prob = float(probs_np[b, i, j])
                truth = int(targets_np[b, i, j])
                pred = 1 if prob > 0.5 else 0

                # Get original DF indices
                idx_i = indices_np[b, i]
                idx_j = indices_np[b, j]

                row_i = val_df.iloc[idx_i]
                row_j = val_df.iloc[idx_j]

                item = {
                    'prob': prob,
                    'txt1': row_i[fc.text], 'amt1': row_i[fc.amount],
                    'txt2': row_j[fc.text], 'amt2': row_j[fc.amount]
                }

                if pred == 1 and truth == 0:
                    fps.append(item)
                elif pred == 0 and truth == 1:
                    fns.append(item)

    logger.info("-" * 50)
    logger.info(f"ADJACENCY MISTAKES (Found {len(fps)} FPs, {len(fns)} FNs)")

    fps_sorted = sorted(fps, key=lambda z: z['prob'], reverse=True)[:num_examples]
    if fps_sorted:
        logger.info("\n>>> Top Adjacency FPs (Model thought these were same):")
        for x in fps_sorted:
            logger.info(f"  [{x['prob']:.4f}] '{x['txt1']}' (${x['amt1']}) <--> '{x['txt2']}' (${x['amt2']})")

    fns_sorted = sorted(fns, key=lambda z: z['prob'])[:num_examples]
    if fns_sorted:
        logger.info("\n>>> Top Adjacency FNs (Model missed these):")
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
    parser.add_argument("--unfreeze_last_n_layers", type=int, default=defaults.unfreeze_last_n_layers)
    parser.add_argument("--epochs", type=int, default=defaults.num_epochs)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--random_state", type=int, default=defaults.random_state)
    parser.add_argument("--downsample", type=float, default=defaults.downsample)
    parser.add_argument("--learning_rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--force_recreate", action="store_true", help="Force recreate data splits")
    parser.add_argument("--text_emb", type=str, default=defaults.text_encoder_model,
                        help=f"Model name. Accepts EmbModel keys (e.g., 'MPNET', 'ALBERT').")
    args = parser.parse_args()

    # Optuna Storage (SQLite for persistence)
    storage_url = f"sqlite:///{args.output_dir}/tuning.db"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Auto-naming logic
    if args.study_name is None:
        study_name = get_next_study_name(storage_url)
    else:
        study_name = args.study_name

    logger.info(f"Using Study Name: {study_name} {get_git_info()} PID={os.getpid()}")

    manager = TuningManager(args, study_name)

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
    if len(study.trials) > 0:
        logger.info(f"Best Trial Score: {study.best_value}")
        logger.info("Best Params:")
        for k, v in study.best_params.items():
            logger.info(f"  {k}: {v}")

        # --- ANALYSIS 1: Hyperparameters ---
        analyze_study_importance(study)

        # --- ANALYSIS 2: Re-Load Best Model for Detailed Analysis ---
        logger.info("=" * 60)
        logger.info("LOADING BEST MODEL FOR DETAILED ANALYSIS")
        logger.info("=" * 60)

        if manager.best_global_path.exists():
            # Create a basic config just to initialize the predictor/analysis tools
            # The saved checkpoint contains the exact config
            checkpoint = torch.load(manager.best_global_path, map_location=torch.device('cpu'), weights_only=False)
            loaded_config = checkpoint['config']
            loaded_config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = TransactionTransformer(loaded_config)
            model.load_state_dict(checkpoint['state_dict'])
            model.to(loaded_config.device)

            # Analyze on Test Set
            analyze_adjacency_mistakes(model, manager.test_df, loaded_config)
            analyze_classification_mistakes(model, manager.test_df, loaded_config)
        else:
            logger.warning("Best model file not found. Skipping analysis.")

    else:
        logger.warning("No trials completed.")


if __name__ == "__main__":
    main()