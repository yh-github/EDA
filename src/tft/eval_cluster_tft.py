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
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
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


def inspect_mistakes(df_prepped, y_true, y_pred, y_probs, field_config, title="Mistakes"):
    print(f"\n>>> EXAMPLES: {title}")

    # Create a DataFrame for easy filtering
    res = df_prepped.copy()
    # We need to map the predictions back to the dataframe rows.
    # Note: df_prepped is 'time-expanded', but predictions are often one per series
    # if predicting the next step. However, TFTRunner output usually aligns with the batch.

    # Important: The 'y_probs' array comes from the *Prediction* step which yields one prob per group
    # (since max_prediction_length=1).
    # We need to map these back to the 'global_group_id's in the dataset.

    # Since we can't easily map row-by-row without the loader's strict order,
    # we will rely on the fact that TFTRunner.predict(return_x=True) gives us the inputs.
    # But for this function, we assume y_probs aligns with the unique groups in the batch order.
    # A safer way for visualization is to rely on the 'x' output from predict.
    pass  # Implemented inside main loop for safer access to 'x' identifiers


def analyze_tune4(model_path: str):
    logger.info(f"Analyzing model at: {model_path}")
    if not Path(model_path).exists():
        logger.error("Model file not found.")
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
        ds = build_clustered_tft_dataset(
            df_p, field_config, meta,
            min_encoder_length=5, max_encoder_length=MAX_ENCODER_LEN
        )
        # We must use 'predict=True' logic via from_dataset to handle TimeSeriesDataSet correctly for inference
        # but build_clustered_tft_dataset returns a fresh dataset.
        # Using the train_ds configuration is safer to ensure encoders match.
        inference_ds = train_ds.from_dataset(train_ds, df_p, predict=True, stop_randomization=True)

        loader = inference_ds.to_dataloader(train=False, batch_size=256, num_workers=4)

        logger.info(f"Predicting on {name}...")

        # Get predictions AND input 'x' to trace back to Group IDs
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
        # Decode Group IDs
        if 'global_group_id' in x['decoder_cat_keys']:  # Newer pytorch-forecasting might differ
            pass

            # Access encoder directly
        group_encoder = train_ds.categorical_encoders['global_group_id']
        # 'global_group_id' index in categorical tensor?
        # We know it's a static categorical, or time-varying known.
        # In 'build_clustered_tft_dataset', it is encoded.
        # usually x['decoder_cat'] contains categoricals.

        # Let's rely on the order.
        # To get the ID, we need to find which column in 'x' corresponds to 'global_group_id'.
        # This is complex in TFT. A simpler way is to index predictions by dataframe if order is preserved.
        # But prediction order usually matches loader order.

        # Try to decode from x['decoder_cat']
        # We need to find the index of 'global_group_id' in known_categoricals + static_categoricals
        # This metadata is in tft.dataset.categorical_encoders

        # Heuristic: We skip complex decoding and just list stats for now,
        # or assume we can't easily link back without custom mapping logic.

        # However, we can calculate FP/FN counts easily.
        fp_mask = (y_pred == 1) & (y_true == 0)
        fn_mask = (y_pred == 0) & (y_true == 1)

        print(f"Mistake Counts:")
        print(f"  False Positives: {fp_mask.sum()}")
        print(f"  False Negatives: {fn_mask.sum()}")

        # To show examples, we'll assume we can grab the 'global_group_id' from the batch 'x'
        # The encoder name is 'global_group_id'.
        try:
            # We need to find the index in x['decoder_cat']
            # tft.dataset.categoricals gives the list of categorical names in order
            cat_names = tft.dataset.categoricals
            if 'global_group_id' in cat_names:
                idx = cat_names.index('global_group_id')
                # decoder_cat shape: [Batch, Time, NumCats]
                group_codes = x['decoder_cat'][:, 0, idx].cpu()
                group_ids = group_encoder.inverse_transform(group_codes)

                # Build mini dataframe for errors
                res_df = pd.DataFrame({
                    'Group': group_ids,
                    'Prob': probs,
                    'True': y_true,
                    'Pred': y_pred
                })

                print("\n>>> False Positive Examples (Predicted Rec, Actual Not):")
                print(res_df[res_df['True'] == 0].sort_values('Prob', ascending=False).head(5).to_string(index=False))

                print("\n>>> False Negative Examples (Predicted Not, Actual Rec):")
                print(res_df[res_df['True'] == 1].sort_values('Prob', ascending=True).head(5).to_string(index=False))

        except Exception as e:
            logger.warning(f"Could not decode Group IDs for examples: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/experiments/analyze_tune4.py <path_to_model.pt>")
        sys.exit(1)

    analyze_tune4(sys.argv[1])