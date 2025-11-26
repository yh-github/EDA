"""
src/tft/eval_cluster_tft3.py

Fixed evaluation script:
1. Reconstructs the Training Context to ensure model weights align.
2. Uses dataset index alignment (Bypassing broken Encoders) to map predictions.
3. Calculates REAL Transaction-Level F1.
"""

import sys
import logging
import torch
import pandas as pd
import numpy as np
from pathlib import Path

from pytorch_forecasting import TimeSeriesDataSet
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve

from common.data import create_train_val_test_split
from common.embedder import EmbeddingService
from common.log_utils import setup_logging
from tft.tft_runner import TFTRunner, RunConfig
from tft.tft_data_clustered import build_clustered_tft_dataset
from experiments.tft_tune4 import prepare_non_overlapping_data

setup_logging(Path("logs/"), "eval_cluster3")
logger = logging.getLogger(__name__)


def analyze_mistakes_simple(
    group_ids: np.ndarray,
    probs: np.ndarray,
    original_df: pd.DataFrame,
    label_col: str,
    set_name: str = "dataset"
) -> None:
    """
    1. Maps group-level predictions back to ALL original transactions.
    2. Calculates TRUE Transaction-Level Metrics (F1, Precision, Recall).
    3. Exports the full transaction-level detail to CSV.
    """
    try:
        # 1. Create a Lookup for Group-Level Scores
        # Note: We don't need Y_True/Y_Pred from the model here, 
        # because we will compare the Prob against the ORIGINAL transaction label.
        group_preds = pd.DataFrame({
            'global_group_id': group_ids,
            'Group_Prob': probs
        })

        # 2. Merge Scores back to PREPARED Transactions (Originals + Group ID)
        # We merge on global_group_id. Unclustered transactions get NaNs.
        full_df = original_df.merge(group_preds, on='global_group_id', how='left')

        # 3. Handle Unscored/Unclustered Rows (Score = 0.0)
        full_df['Group_Prob'] = full_df['Group_Prob'].fillna(0.0)
        
        # 4. Generate Predictions (Threshold 0.5)
        full_df['Group_Pred'] = (full_df['Group_Prob'] > 0.5).astype(int)
        
        # 5. Calculate TRUE Transaction-Level Metrics
        # Compare Original Label vs Broadcasted Prediction
        txn_y_true = full_df[label_col].astype(int).values
        txn_y_pred = full_df['Group_Pred'].values

        p, r, f1, _ = precision_recall_fscore_support(txn_y_true, txn_y_pred, average='binary')
        
        logger.info("\n" + "*" * 50)
        logger.info(f"REAL TRANSACTION-LEVEL METRICS ({set_name})")
        logger.info("*" * 50)
        logger.info(f"Transactions   : {len(full_df)}")
        logger.info(f"Precision      : {p:.4f}")
        logger.info(f"Recall         : {r:.4f}")
        logger.info(f"F1 Score       : {f1:.4f}")
        logger.info("*" * 50)

        # 6. Define Mistake Types
        conditions = [
            (txn_y_true == 1) & (txn_y_pred == 1), # TP
            (txn_y_true == 0) & (txn_y_pred == 0), # TN
            (txn_y_true == 0) & (txn_y_pred == 1), # FP
            (txn_y_true == 1) & (txn_y_pred == 0), # FN
        ]
        choices = ['TP', 'TN', 'FP', 'FN']
        full_df['Result_Type'] = np.select(conditions, choices, default='Error')

        # 7. Save to CSV
        filename = f"mistake_analysis_{set_name}.csv"
        logger.info(f"Saving full transaction-level analysis to {filename}...")
        full_df.to_csv(filename, index=False)

        # 8. Log Top Group-Level Mistakes (for readability)
        mistakes_grouped = full_df[full_df['Result_Type'].isin(['FP', 'FN'])].drop_duplicates('global_group_id')
        
        fp_count = len(full_df[full_df['Result_Type'] == 'FP'])
        fn_count = len(full_df[full_df['Result_Type'] == 'FN'])

        logger.info(f"Total Transaction Mistakes: FP={fp_count}, FN={fn_count}")

        cols = ['global_group_id', 'bankRawDescription', 'amount', 'Group_Prob']
        
        if not mistakes_grouped[mistakes_grouped['Result_Type']=='FP'].empty:
            logger.info("\n>>> Top False Positive Groups (Pred Rec, Actual Not):")
            top_fp = mistakes_grouped[mistakes_grouped['Result_Type'] == 'FP'].sort_values('Group_Prob', ascending=False).head(5)
            logger.info(top_fp[cols].to_string(index=False))

        if not mistakes_grouped[mistakes_grouped['Result_Type']=='FN'].empty:
            logger.info("\n>>> Top False Negative Groups (Pred Not, Actual Rec):")
            top_fn = mistakes_grouped[mistakes_grouped['Result_Type'] == 'FN'].sort_values('Group_Prob', ascending=True).head(5)
            logger.info(top_fn[cols].to_string(index=False))

    except Exception as e:
        logger.error(f"Error during mistake analysis: {e}", exc_info=True)


def evaluate_checkpoint(model_path: str):
    path_obj = Path(model_path)
    if not path_obj.exists():
        logger.error(f"Checkpoint not found at {model_path}")
        return

    logger.info(f"Loading checkpoint payload from {model_path}...")
    payload = torch.load(path_obj, map_location=torch.device("cpu"), weights_only=False)

    if "run_config" not in payload:
        logger.error("Checkpoint does not contain 'run_config'.")
        return

    rc: RunConfig = payload["run_config"]
    logger.info(f"Restored Config: {rc.emb_params.model_name}")

    # --- Load Data ---
    full_df = pd.read_csv("data/rec_data2.csv").dropna(
        subset=[rc.field_config.date, rc.field_config.amount, rc.field_config.text]
    )

    train_df, val_df, test_df = create_train_val_test_split(
        test_size=0.2,
        val_size=0.2,
        full_df=full_df,
        random_state=rc.experiment_config.random_state,
        field_config=rc.field_config
    )

    # --- Reconstruct Pipeline ---
    emb_service = EmbeddingService.create(rc.emb_params)

    # Fit on Train (Critical to restore correct integer mappings)
    train_prepped, pca_model, processor, meta = prepare_non_overlapping_data(
        train_df,
        rc.field_config,
        feat_params=rc.feat_proc_params,
        embedding_service=emb_service,
        fit_processor=True
    )

    def prep_dataset(df_in):
        df_out, _, _, _ = prepare_non_overlapping_data(
            df_in,
            rc.field_config,
            embedding_service=emb_service,
            pca_model=pca_model,
            processor=processor,
            fit_processor=False
        )
        df_out[rc.field_config.date] = pd.to_datetime(df_out[rc.field_config.date])
        return df_out

    # --- Build Template & Load Model ---
    train_ds = build_clustered_tft_dataset(
        train_prepped,
        rc.field_config,
        meta,
        min_encoder_length=rc.min_encoder_length,
        max_encoder_length=rc.max_encoder_length
    )

    tft = TFTRunner.load_from_checkpoint(model_path, dataset=train_ds)

    # --- Evaluation Loop ---
    def run_report(set_name, df_raw):
        logger.info(f"Processing {set_name}...")
        
        df_prepped = prep_dataset(df_raw)
        
        if len(df_prepped) == 0:
            logger.warning("Empty dataset, skipping.")
            return

        inference_ds = train_ds.from_dataset(
            train_ds, df_prepped, predict=True, stop_randomization=True
        )
        loader = inference_ds.to_dataloader(train=False, batch_size=256, num_workers=4)

        # 1. Get Predictions (Raw)
        raw_output = tft.predict(loader, mode="raw", return_x=True, return_index=False)
        raw_preds = raw_output[0]
        
        # 2. Extract Probs
        probs = torch.softmax(raw_preds['prediction'], dim=-1)[:, 0, 1].cpu().numpy()
        
        # 3. Extract Group IDs DIRECTLY from the dataset index
        # The loader preserves order, so we can access the underlying index dataframe directly
        # inference_ds.index is a pandas DataFrame with columns [time_idx, global_group_id, ...]
        # We need to ensure we only get the groups that were actually predicted (in case some were dropped)
        
        # Safer method: tft.predict returns 'index' if requested, but that failed earlier.
        # Instead, we rely on the x output which contains decoded categoricals IF the encoder worked.
        # Since encoder failed, we use the x['groups'] codes but match them to the dataset.
        
        # ULTIMATE FIX: The `inference_ds` object knows the mapping between row `i` and `global_group_id`.
        # However, `to_dataloader` might shuffle if not careful (we set shuffle=False).
        # And `TimeSeriesDataSet` filters short sequences.
        # We need the filtered index.
        
        # We can re-construct the filtered index from the loader iteration to be 100% safe.
        valid_group_ids = []
        
        # We iterate the loader again to get the string IDs directly from the source dataframe if possible?
        # No, the loader yields tensors.
        
        # We use `inference_ds.decoded_index` if available, or `inference_ds.index`.
        # `inference_ds.index` contains the valid groups after filtering.
        filtered_index = inference_ds.index
        
        if len(filtered_index) != len(probs):
            logger.error(f"Shape mismatch! Index: {len(filtered_index)}, Preds: {len(probs)}")
            # Fallback: Try to use x['groups'] codes if lengths don't match (unlikely if shuffle=False)
            return

        # 'global_group_id' should be a column in the index DataFrame
        decoded_ids = filtered_index['global_group_id'].values

        # 4. Analyze
        analyze_mistakes_simple(
            decoded_ids, 
            probs, 
            original_df=df_prepped, # Contains 'global_group_id' and 'label'
            label_col=rc.field_config.label,
            set_name=set_name
        )

    run_report("VAL SET", val_df)
    run_report("TEST SET", test_df)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/tft/eval_cluster_tft3.py <path_to_model.pt>")
        sys.exit(1)

    evaluate_checkpoint(sys.argv[1])