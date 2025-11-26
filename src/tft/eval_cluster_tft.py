#eval tft_tune4

import sys
import logging
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve

from common.config import FieldConfig, EmbModel, ExperimentConfig
from common.data import create_train_val_test_split
from common.embedder import EmbeddingService
from common.feature_processor import FeatProcParams
from common.log_utils import setup_logging
from tft.tft_runner import TFTRunner
from tft.tft_data_clustered import build_clustered_tft_dataset

# Import the specific preparation logic from the training script
from experiments.tft_tune4 import prepare_non_overlapping_data

setup_logging(Path("logs/"), "analyze_tune4")
logger = logging.getLogger(__name__)


def print_metrics(y_true, y_pred, y_probs, set_name="DataSet"):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    # Calculate Best F1 / Threshold
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    # Add epsilon to avoid division by zero
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)

    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    # thresholds array is 1 shorter than prec/rec
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 1.0

    print("\n" + "=" * 50)
    print(f"METRICS: {set_name}")
    print("=" * 50)
    print(f"Threshold Used : 0.5")
    print(f"Precision      : {p:.4f}")
    print(f"Recall         : {r:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print("-" * 30)
    print(f"Best Potential F1 : {best_f1:.4f} (at threshold {best_thresh:.4f})")
    print("=" * 50)


def analyze_tune4(model_path: str):
    logger.info(f"Analyzing model at: {model_path}")
    if not Path(model_path).exists():
        logger.error(f"Model file not found at {model_path}")
        return

    # 1. Configs
    field_config = FieldConfig()
    exp_config = ExperimentConfig()

    # Match the feat_params used in TFTTuningExperiment default
    feat_params = FeatProcParams(
        use_is_positive=False,
        use_categorical_dates=True,
        use_cyclical_dates=True,
        use_continuous_amount=True,
        use_categorical_amount=False,
        k_top=0,
        n_bins=0
    )

    # 2. Load and Split Data
    logger.info("Loading data...")
    full_df = pd.read_csv("data/rec_data2.csv").dropna(
        subset=[field_config.date, field_config.amount, field_config.text]
    )

    # Reproduce the exact split
    train_df, val_df, test_df = create_train_val_test_split(
        test_size=0.2,
        val_size=0.2,
        full_df=full_df,
        random_state=exp_config.random_state
    )

    logger.info(f"Data Sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 3. Fit Pipeline on TRAIN
    logger.info("Fitting processing pipeline on TRAIN set...")
    emb_service = EmbeddingService(model_name=EmbModel.MPNET, max_length=64, batch_size=512)

    # prepare_non_overlapping_data fits the processor and PCA
    train_prepped, pca_model, processor, meta = prepare_non_overlapping_data(
        train_df, field_config, feat_params=feat_params,
        embedding_service=emb_service, fit_processor=True
    )

    # 4. Transform VAL and TEST
    logger.info("Transforming VAL set...")
    val_prepped, _, _, _ = prepare_non_overlapping_data(
        val_df, field_config, embedding_service=emb_service,
        pca_model=pca_model, processor=processor, fit_processor=False
    )

    logger.info("Transforming TEST set...")
    test_prepped, _, _, _ = prepare_non_overlapping_data(
        test_df, field_config, embedding_service=emb_service,
        pca_model=pca_model, processor=processor, fit_processor=False
    )

    # 5. Build Datasets (Template for Model Loading)
    # Tune4 used max_encoder_len = 64
    MAX_ENCODER_LEN = 64

    # We need a dummy dataset to initialize the model structure
    logger.info("Building Template Dataset...")
    train_ds = build_clustered_tft_dataset(
        train_prepped, field_config, meta,
        min_encoder_length=5, max_encoder_length=MAX_ENCODER_LEN
    )

    # 6. Load Model
    logger.info("Loading Checkpoint...")
    tft = TFTRunner.load_from_checkpoint(model_path, dataset=train_ds)

    # 7. Evaluate Loop
    for name, df_p in [("VALIDATION", val_prepped), ("TEST", test_prepped)]:
        if df_p.empty:
            logger.warning(f"Skipping {name} (Empty DataFrame)")
            continue

        # Create Loader
        # We use train_ds.from_dataset to ensure encoders/scalers are copied correctly
        inference_ds = train_ds.from_dataset(
            train_ds, df_p, predict=True, stop_randomization=True
        )

        loader = inference_ds.to_dataloader(train=False, batch_size=256, num_workers=4)

        logger.info(f"Predicting on {name}...")

        # Get predictions AND input 'x'
        raw_output = tft.predict(loader, mode="raw", return_x=True)
        raw_preds = raw_output[0]  # dictionary with 'prediction'
        x = raw_output[1]

        # Extract Probabilities (Class 1)
        # Shape: [Batch, 1, 2]
        probs = torch.softmax(raw_preds['prediction'], dim=-1)[:, 0, 1].cpu().numpy()
        y_true = x['decoder_target'][:, 0].cpu().numpy()

        y_pred = (probs > 0.5).astype(int)

        # Metrics
        print_metrics(y_true, y_pred, probs, set_name=name)

        # --- MISTAKE ANALYSIS ---
        try:
            # FIX: Resolve feature index using the dataset metadata
            cat_names = tft.dataset.categoricals
            target_group_col = 'global_group_id'

            if target_group_col in cat_names:
                idx = cat_names.index(target_group_col)

                # Retrieve encoder
                group_encoder = tft.dataset.categorical_encoders[target_group_col]

                # decoder_cat shape: [Batch, Time, NumCats]
                # We take time=0 since the group is static/constant for the sample
                group_codes = x['decoder_cat'][:, 0, idx].cpu()

                # Inverse transform to get string IDs
                group_ids = group_encoder.inverse_transform(group_codes)

                # Build mini dataframe for errors
                res_df = pd.DataFrame({
                    'Group': group_ids,
                    'Prob': probs,
                    'True': y_true,
                    'Pred': y_pred
                })

                fp_df = res_df[(res_df['True'] == 0) & (res_df['Pred'] == 1)]
                fn_df = res_df[(res_df['True'] == 1) & (res_df['Pred'] == 0)]

                print(f"Mistake Counts (Total Series: {len(res_df)}):")
                print(f"  False Positives: {len(fp_df)}")
                print(f"  False Negatives: {len(fn_df)}")

                if not fp_df.empty:
                    print("\n>>> False Positive Examples (Predicted Rec, Actual Not):")
                    # Show top confident mistakes
                    print(fp_df.sort_values('Prob', ascending=False).head(5).to_string(index=False))

                if not fn_df.empty:
                    print("\n>>> False Negative Examples (Predicted Not, Actual Rec):")
                    # Show top confident mistakes (Prob close to 0)
                    print(fn_df.sort_values('Prob', ascending=True).head(5).to_string(index=False))

            else:
                logger.warning(f"'{target_group_col}' not found in dataset categoricals: {cat_names}")

        except Exception as e:
            logger.error(f"Error decoding examples: {e}", exc_info=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/experiments/analyze_tune4.py <path_to_model.pt>")
        sys.exit(1)

    analyze_tune4(sys.argv[1])