"""
src/tft/eval_cluster2.py

A simplified evaluation script for TFT models trained with the new Checkpoint format
(which includes RunConfig).

It automatically loads:
1. Embedding Parameters
2. Feature Processing Parameters
3. Model Hyperparameters

Designed specifically for models like tft_tune5 / tft_tune6 which use:
- Non-Overlapping Clustering
- Clustered Dataset Building
"""

import sys
import logging
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve

from common.data import create_train_val_test_split
from common.embedder import EmbeddingService
from common.log_utils import setup_logging
from tft.tft_runner import TFTRunner, RunConfig
from tft.tft_data_clustered import build_clustered_tft_dataset

# Import the specific preparation logic used in Tune 5/6
from experiments.tft_tune4 import prepare_non_overlapping_data

setup_logging(Path("logs/"), "eval_cluster2")
logger = logging.getLogger(__name__)


def print_metrics(y_true, y_pred, y_probs, set_name="DataSet"):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    # Calculate Best F1 / Threshold
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)

    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    # Thresholds is 1 shorter than precision/recall
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


def analyze_mistakes(tft, x, probs, y_true, y_pred):
    """
    Decodes the Group IDs to show specific false positives/negatives.
    """
    try:
        # 1. Identify Group ID Column from encoders
        # In tune5/6 this is 'global_group_id'
        target_group_col = 'global_group_id'

        if target_group_col not in tft.dataset.categorical_encoders:
            logger.warning(f"Could not find '{target_group_col}' in dataset encoders. Skipping mistake analysis.")
            return

        group_encoder = tft.dataset.categorical_encoders[target_group_col]
        cat_names = tft.dataset.categoricals
        idx = cat_names.index(target_group_col)

        # 2. Extract IDs
        # decoder_cat shape: [Batch, Time, NumCats]
        # Groups are static, so Time=0 is fine
        group_codes = x['decoder_cat'][:, 0, idx].cpu()
        group_ids = group_encoder.inverse_transform(group_codes)

        # 3. Build Analysis DataFrame
        res_df = pd.DataFrame({
            'Group': group_ids,
            'Prob': probs,
            'True': y_true,
            'Pred': y_pred
        })

        fp_df = res_df[(res_df['True'] == 0) & (res_df['Pred'] == 1)]
        fn_df = res_df[(res_df['True'] == 1) & (res_df['Pred'] == 0)]

        print(f"\nMistake Counts (Total Series: {len(res_df)}):")
        print(f"  False Positives: {len(fp_df)}")
        print(f"  False Negatives: {len(fn_df)}")

        if not fp_df.empty:
            print("\n>>> False Positive Examples (Predicted Rec, Actual Not):")
            print(fp_df.sort_values('Prob', ascending=False).head(5).to_string(index=False))

        if not fn_df.empty:
            print("\n>>> False Negative Examples (Predicted Not, Actual Rec):")
            print(fn_df.sort_values('Prob', ascending=True).head(5).to_string(index=False))

    except Exception as e:
        logger.error(f"Error during mistake analysis: {e}")


def evaluate_checkpoint(model_path: str):
    path_obj = Path(model_path)
    if not path_obj.exists():
        logger.error(f"Checkpoint not found at {model_path}")
        return

    logger.info(f"Loading checkpoint payload from {model_path}...")
    # Load CPU mapped to avoid GPU errors if running locally
    payload = torch.load(path_obj, map_location=torch.device("cpu"))

    # --- 1. Extract RunConfig ---
    # This contains all the setup used during training!
    if "run_config" not in payload:
        logger.error("Checkpoint does not contain 'run_config'. Cannot use eval_cluster2.py.")
        return

    rc: RunConfig = payload["run_config"]

    logger.info("Restored Configuration:")
    logger.info(f"  Embedder Model: {rc.emb_params.model_name}")
    logger.info(f"  Feat Params: {rc.feat_proc_params}")

    # --- 2. Load and Split Data ---
    logger.info("Loading Data...")
    full_df = pd.read_csv("data/rec_data2.csv").dropna(
        subset=[rc.field_config.date, rc.field_config.amount, rc.field_config.text]
    )

    # Use the random state from the saved config to reproduce the exact split
    train_df, val_df, test_df = create_train_val_test_split(
        test_size=0.2,
        val_size=0.2,
        full_df=full_df,
        random_state=rc.experiment_config.random_state,
        field_config=rc.field_config
    )

    logger.info(f"Split Sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # --- 3. Fit Pipeline on Train (Reconstruction) ---
    logger.info("Reconstructing Pipeline (Fitting on Train)...")

    # Create Embedder
    emb_service = EmbeddingService.create(rc.emb_params)

    # Prepare Train Data (Fits Processor & PCA)
    # NOTE: We use prepare_non_overlapping_data because this script is for Tune5/6
    train_prepped, pca_model, processor, meta = prepare_non_overlapping_data(
        train_df,
        rc.field_config,
        feat_params=rc.feat_proc_params,
        embedding_service=emb_service,
        fit_processor=True
    )

    # --- 4. Transform Test Data ---
    logger.info("Transforming Test Data...")
    test_prepped, _, _, _ = prepare_non_overlapping_data(
        test_df,
        rc.field_config,
        embedding_service=emb_service,
        pca_model=pca_model,
        processor=processor,
        fit_processor=False
    )

    # Ensure dates are datetime (Safety fix)
    test_prepped[rc.field_config.date] = pd.to_datetime(test_prepped[rc.field_config.date])

    # --- 5. Build Dataset Template ---
    # We need a dataset to initialize the model.
    # Tune 5/6 used max_encoder_len=64.
    logger.info("Building Template Dataset...")
    train_ds = build_clustered_tft_dataset(
        train_prepped,
        rc.field_config,
        meta,
        min_encoder_length=5,
        max_encoder_length=64  # Default for Tune 5/6
    )

    # --- 6. Load Model ---
    logger.info("Loading Model Weights...")
    # TFTRunner.load_from_checkpoint handles the architecture reconstruction
    tft = TFTRunner.load_from_checkpoint(model_path, dataset=train_ds)

    # --- 7. Prediction Loop ---
    logger.info("Running Predictions on Test Set...")

    # Create Inference Dataset from Template
    inference_ds = train_ds.from_dataset(
        train_ds, test_prepped, predict=True, stop_randomization=True
    )

    loader = inference_ds.to_dataloader(train=False, batch_size=256, num_workers=4)

    # Predict
    raw_output = tft.predict(loader, mode="raw", return_x=True)
    raw_preds = raw_output[0]
    x = raw_output[1]

    # Extract Probabilities (Class 1 = Recurring)
    probs = torch.softmax(raw_preds['prediction'], dim=-1)[:, 0, 1].cpu().numpy()
    y_true = x['decoder_target'][:, 0].cpu().numpy()
    y_pred = (probs > 0.5).astype(int)

    # --- 8. Results ---
    print_metrics(y_true, y_pred, probs, set_name="TEST SET")

    analyze_mistakes(tft, x, probs, y_true, y_pred)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/tft/eval_cluster2.py <path_to_model.pt>")
        sys.exit(1)

    evaluate_checkpoint(sys.argv[1])
