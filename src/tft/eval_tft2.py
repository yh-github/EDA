import torch
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from sklearn.metrics import precision_recall_fscore_support
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from common.config import FieldConfig, EmbModel
from tft.tft_data import prepare_tft_data, build_tft_dataset
from common.data import create_train_val_test_split
from common.log_utils import setup_logging
from tft.tft_runner import TFTRunner
from common.feature_processor import FeatProcParams
from common.embedder import EmbeddingService

setup_logging(Path("logs/"), "eval_refined")
logger = logging.getLogger(__name__)

@dataclass
class InteroperableGroup:
    group_id: str
    description: str
    amount_stats: float
    cycle_days: float
    confidence: float
    is_recurring: bool


class RobustGroupExtractor:
    def __init__(self, model: TemporalFusionTransformer, df: pd.DataFrame, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold

        # 1. Auto-detect the group key
        if 'global_group_id' in model.dataset.categorical_encoders:
            self.group_key = 'global_group_id'
        elif 'accountId' in model.dataset.categorical_encoders:
            self.group_key = 'accountId'
        else:
            raise ValueError("Could not find 'global_group_id' or 'accountId' in model encoders.")

        self.group_encoder = model.dataset.categorical_encoders[self.group_key]

        # 2. FORCE REPAIR: Always fit the encoder to ensure internal state exists
        # This fixes the "AttributeError: no attribute 'classes_vector_'"
        try:
            self.group_encoder.fit(df[self.group_key].astype(str))
        except Exception as e:
            logger.warning(f"Encoder fit warning: {e}")

        logger.info(f"Extractor initialized using group key: {self.group_key}")

    def extract(self, raw_predictions, x, original_df):
        results = []

        # 1. Decode Group IDs
        encoded_group_ids = x['decoder_cat'][:, 0, 0].cpu().numpy()
        decoded_group_ids = self.group_encoder.inverse_transform(encoded_group_ids)

        # 2. Extract Mean Probabilities
        recurrence_probs = torch.softmax(raw_predictions, dim=-1)[:, :, 1]
        mean_probs = recurrence_probs.mean(dim=1).cpu().numpy()

        # 3. Group the DataFrame for fast lookup
        if self.group_key not in original_df.columns:
            raise KeyError(f"DataFrame is missing the group column: {self.group_key}")

        df_grouped = original_df.groupby(self.group_key)

        for group_id, confidence in zip(decoded_group_ids, mean_probs):
            if group_id not in df_grouped.groups:
                continue

            group_rows = df_grouped.get_group(group_id)

            # Metadata
            text_mode = group_rows['bankRawDescription'].mode()
            description = text_mode[0] if not text_mode.empty else "Unknown"
            median_amount = float(group_rows['amount'].median())

            # Cycle Calculation
            # We assume 'date' is already converted to datetime by the caller
            dates = group_rows['date'].sort_values()
            if len(dates) > 1:
                diffs = dates.diff().dt.days.dropna()
                cycle_days = float(diffs.median())
            else:
                cycle_days = 0.0

            results.append(InteroperableGroup(
                group_id=str(group_id),
                description=str(description),
                amount_stats=median_amount,
                cycle_days=cycle_days,
                confidence=float(confidence),
                is_recurring=bool(confidence >= self.threshold)
            ))

        return results


def is_simple_cycle(days: float) -> bool:
    """
    Returns True only for Weekly (~7), Bi-Weekly (~14), or Monthly (~30) cycles.
    """
    if days < 5: return False  # Too frequent (daily/bursts)

    # Weekly: 7 +/- 2 days (5 to 9)
    if 5 <= days <= 9: return True

    # Bi-Weekly: 14 +/- 2 days (12 to 16)
    if 12 <= days <= 16: return True

    # Monthly: 30 +/- 3 days (27 to 33)
    if 27 <= days <= 33: return True

    return False


def evaluate_refined(mode_path:Path):
    # 1. Load Data
    field_config = FieldConfig()
    full_df = pd.read_csv("data/rec_data2.csv").dropna(
        subset=[field_config.date, field_config.amount, field_config.text]
    )

    _, val_df, unseen_test_df = create_train_val_test_split(test_size=0.2, val_size=0.2, full_df=full_df, random_state=112025)

    def do_eval(df_name:str, test_df:pd.DataFrame):
        logger.info("Preparing test data...")

        feat_params = FeatProcParams(
            use_is_positive=False,
            use_categorical_dates=True,
            use_cyclical_dates=True,
            use_continuous_amount=True,
            use_categorical_amount=False,
            k_top=0,
            n_bins=0
        )

        # Initialize Embedder
        emb_service = EmbeddingService(model_name=EmbModel.MPNET, max_length=64, batch_size=256)

        # Prepare Data
        test_df_prepped, _, _, meta = prepare_tft_data(
            test_df,
            field_config,
            feat_params=feat_params,
            embedding_service=emb_service,
            fit_processor=True
        )

        # --- CRITICAL FIX: Ensure Dates are Datetime Objects ---
        test_df_prepped[field_config.date] = pd.to_datetime(test_df_prepped[field_config.date])

        # 2. Build Dataset Template
        dummy_ds = build_tft_dataset(
            test_df_prepped,
            field_config,
            meta,
            min_encoder_length=10,
            max_encoder_length=150
        )

        # 3. Load Model
        if not Path(mode_path).exists():
            logger.error(f"Checkpoint not found at {mode_path}")
            return

        tft = TFTRunner.load_from_checkpoint(mode_path, dataset=dummy_ds)
        tft.dataset = dummy_ds
        logger.info("Model loaded successfully.")

        # 4. Create Loader
        test_ds = TimeSeriesDataSet.from_dataset(dummy_ds, test_df_prepped, predict=True, stop_randomization=True)
        test_loader = test_ds.to_dataloader(train=False, batch_size=256, num_workers=0)

        # 5. Raw Predictions
        logger.info("Running raw predictions...")
        prediction_output = tft.predict(test_loader, mode="raw", return_x=True)
        raw_predictions = prediction_output[0]
        x = prediction_output[1]

        y_prob = torch.softmax(raw_predictions['prediction'], dim=-1)[:, 0, 1].cpu().numpy()
        y_true = x["decoder_target"][:, 0].cpu().numpy()

        # --- METRICS: RAW ---
        y_pred_raw = (y_prob > 0.5).astype(int)
        p_raw, r_raw, f1_raw, _ = precision_recall_fscore_support(y_true, y_pred_raw, average='binary')

        logger.info("\n" + "=" * 40)
        logger.info(f"BEFORE (Raw Model @ 0.5)")
        logger.info(f"Precision: {p_raw:.4f} | Recall: {r_raw:.4f} | F1: {f1_raw:.4f}")
        logger.info("=" * 40)

        def runGroupExtractor():
            logger.info("Running RobustGroupExtractor...")
            extractor = RobustGroupExtractor(model=tft, df=test_df_prepped, threshold=0.5)
            groups = extractor.extract(raw_predictions['prediction'], x, test_df_prepped)

            # 7. Apply Refinement Rules
            group_decisions = {}

            rejected_reasons = {"low_conf": 0, "bad_cycle": 0}

            for g in groups:
                # STEP A: Use generous threshold (0.4) to catch everything possible
                is_rec = (g.confidence > 0.40)

                if not is_rec:
                    rejected_reasons["low_conf"] += 1
                else:
                    # STEP B: Strict Simple Cycle Check (7, 14, 30 days)
                    if not is_simple_cycle(g.cycle_days):
                        is_rec = False
                        rejected_reasons["bad_cycle"] += 1

                group_decisions[g.group_id] = 1 if is_rec else 0

            logger.info(
                f"Rejection Stats: Low Conf: {rejected_reasons['low_conf']}, Bad Cycle: {rejected_reasons['bad_cycle']}")

            # Map back
            group_encoder = extractor.group_encoder
            batch_group_codes = x['decoder_cat'][:, 0, 0].cpu()
            batch_group_ids = group_encoder.inverse_transform(batch_group_codes)

            y_pred_refined = []
            for gid in batch_group_ids:
                y_pred_refined.append(group_decisions.get(gid, 0))

            y_pred_refined = np.array(y_pred_refined)

            # --- METRICS: REFINED ---
            p_ref, r_ref, f1_ref, _ = precision_recall_fscore_support(y_true, y_pred_refined, average='binary')

            logger.info("\n" + "=" * 40)
            logger.info(f"AFTER (With Extractor & Simple Cycles)")
            logger.info(f"Precision: {p_ref:.4f} | Recall: {r_ref:.4f} | F1: {f1_ref:.4f}")
            logger.info("=" * 40)
            logger.info(f"CHANGE: P: {p_ref - p_raw:+.4f} | R: {r_ref - r_raw:+.4f} | F1: {f1_ref - f1_raw:+.4f}")

            # 8. Examples
            saved_fp_indices = np.where((y_pred_raw == 1) & (y_pred_refined == 0) & (y_true == 0))[0]
            if len(saved_fp_indices) > 0:
                logger.info(f"\nCaught {len(saved_fp_indices)} False Positives! Examples:")
                filtered_groups = [g for g in groups if g.confidence > 0.65 and group_decisions[g.group_id] == 0]
                filtered_groups.sort(key=lambda x: x.confidence, reverse=True)
                for g in filtered_groups[:5]:
                    logger.info(f"  - Rejected '{g.description}' (Conf: {g.confidence:.2f}, Cycle: {g.cycle_days:.1f} days)")

    do_eval("val_df", val_df)
    do_eval("unseen_test_df", unseen_test_df)

import sys
if __name__ == "__main__":
    evaluate_refined(Path(sys.argv[1]))