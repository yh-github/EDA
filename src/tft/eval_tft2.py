import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from sklearn.metrics import precision_recall_fscore_support

from common.config import FieldConfig
from common.data import create_train_val_test_split
from common.feature_processor import FeatProcParams
from common.log_utils import setup_logging
from tft.group import TFTGroupExtractor
from tft.tft_data import prepare_tft_data, build_tft_dataset
from tft.tft_runner import TFTRunner  # <--- Import TFTRunner

setup_logging(Path("logs/"), "eval_refined")
logger = logging.getLogger(__name__)

# UPDATE THIS TO YOUR MODEL PATH
CHECKPOINT_PATH = "cache/tft_models/best_tune1.1_model.pt"


def evaluate_refined():
    # 1. Load Data
    field_config = FieldConfig()
    full_df = pd.read_csv("data/rec_data2.csv").dropna(
        subset=[field_config.date, field_config.amount, field_config.text]
    )

    _, _, test_df = create_train_val_test_split(test_size=0.2, val_size=0.2, full_df=full_df, random_state=112025)

    logger.info("Preparing test data...")

    # --- CONFIG ---
    # FIXME Re-use the params from training (src/tft/tft_experiment.py)
    feat_params = FeatProcParams(
        use_is_positive=False,
        use_categorical_dates=True,
        use_cyclical_dates=True,
        use_continuous_amount=True,
        use_categorical_amount=False,
        k_top=0,
        n_bins=0
    )

    # Prepare Data with fit_processor=True to regenerate metadata/vocab
    test_df_prepped, _, _, meta = prepare_tft_data(
        test_df,
        field_config,
        feat_params=feat_params,
        fit_processor=True
    )

    # 2. Build Dataset Structure (Empty, just for model initialization)
    # Note: Encoder length should match training or be sufficient for test data
    # We use a safe default here or match training config (e.g. 10 to 64)
    dummy_ds = build_tft_dataset(
        test_df_prepped,
        field_config,
        meta,
        min_encoder_length=5,
        max_encoder_length=64
    )

    # 3. Load Model using custom loader
    if not Path(CHECKPOINT_PATH).exists():
        logger.error(f"Checkpoint not found at {CHECKPOINT_PATH}")
        return

    # Use TFTRunner to load the custom checkpoint format
    tft = TFTRunner.load_from_checkpoint(CHECKPOINT_PATH, dataset=dummy_ds)
    logger.info("Model loaded successfully via TFTRunner.")

    # 4. Create Test Loader
    test_ds = TimeSeriesDataSet.from_dataset(tft.dataset, test_df_prepped, predict=True, stop_randomization=True)
    test_loader = test_ds.to_dataloader(train=False, batch_size=256, num_workers=4)

    # 5. Raw Predictions
    logger.info("Running raw predictions...")
    raw_predictions, x = tft.predict(test_loader, mode="raw", return_x=True)

    y_prob = torch.softmax(raw_predictions['prediction'], dim=-1)[:, 0, 1].cpu().numpy()
    y_true = x["decoder_target"][:, 0].cpu().numpy()

    # --- METRIC SET 1: RAW MODEL ---
    y_pred_raw = (y_prob > 0.5).astype(int)
    p_raw, r_raw, f1_raw, _ = precision_recall_fscore_support(y_true, y_pred_raw, average='binary')

    logger.info("\n" + "=" * 40)
    logger.info(f"BEFORE (Raw Model @ 0.5)")
    logger.info(f"Precision: {p_raw:.4f} | Recall: {r_raw:.4f} | F1: {f1_raw:.4f}")
    logger.info("=" * 40)

    # 6. Run Extractor (The "Business Layer")
    logger.info("Running TFTGroupExtractor...")
    extractor = TFTGroupExtractor(model=tft, threshold=0.5)
    groups = extractor.extract(raw_predictions['prediction'], x, test_df_prepped)

    # 7. Apply Refinement Rules
    group_decisions = {}
    for g in groups:
        is_rec = (g.confidence > 0.65)
        if is_rec:
            # Rule: Must have a plausible cycle (approx weekly or monthly)
            valid_cycle = (6 <= g.cycle_days <= 35)
            if not valid_cycle:
                is_rec = False
        group_decisions[g.group_id] = 1 if is_rec else 0

    # Map back to transactions
    group_encoder = tft.dataset.categorical_encoders[field_config.accountId]  # Standard uses accountId as group
    # Note: If using clustered data, this would be 'global_group_id'.
    # Since prepare_tft_data (standard) uses accountId as group_ids, we stick to that.

    # Check what group key was used in build_tft_dataset.
    # Standard build_tft_dataset uses group_ids=[field_config.accountId]

    batch_group_codes = x['decoder_cat'][:, 0, 0].cpu()
    batch_group_ids = group_encoder.inverse_transform(batch_group_codes)

    y_pred_refined = []
    for gid in batch_group_ids:
        y_pred_refined.append(group_decisions.get(gid, 0))

    y_pred_refined = np.array(y_pred_refined)

    # --- METRIC SET 2: REFINED ---
    p_ref, r_ref, f1_ref, _ = precision_recall_fscore_support(y_true, y_pred_refined, average='binary')

    logger.info("\n" + "=" * 40)
    logger.info(f"AFTER (With Extractor & Rules)")
    logger.info(f"Precision: {p_ref:.4f} | Recall: {r_ref:.4f} | F1: {f1_ref:.4f}")
    logger.info("=" * 40)
    logger.info(f"CHANGE: P: {p_ref - p_raw:+.4f} | R: {r_ref - r_raw:+.4f} | F1: {f1_ref - f1_raw:+.4f}")


if __name__ == "__main__":
    evaluate_refined()