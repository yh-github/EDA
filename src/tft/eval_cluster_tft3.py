"""
src/tft/eval_cluster3.py

Fixed evaluation script:
1. Passes PREPARED data (with global_group_id) to analysis to avoid KeyErrors.
2. Improved robust_decode_groups to handle different Encoder versions.
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


def robust_decode_groups(encoder, codes: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Manually decodes group IDs using the encoder's classes dictionary.
    Handles different PyTorch Forecasting encoder versions.
    """
    if isinstance(codes, torch.Tensor):
        codes = codes.cpu().numpy().astype(int)
    
    # Attempt 1: Standard inverse_transform
    try:
        return encoder.inverse_transform(codes)
    except Exception:
        pass

    # Attempt 2: classes_ dictionary (Label -> Int)
    if hasattr(encoder, "classes_") and isinstance(encoder.classes_, dict):
        int_to_label = {v: k for k, v in encoder.classes_.items()}
        return np.array([int_to_label.get(c, "Unknown") for c in codes])

    # Attempt 3: classes_ list/array (Index -> Label)
    if hasattr(encoder, "classes_") and (isinstance(encoder.classes_, list) or isinstance(encoder.classes_, np.ndarray)):
        vocab = encoder.classes_
        decoded = []
        for c in codes:
            if c < len(vocab):
                decoded.append(vocab[c])
            else:
                decoded.append("Unknown")
        return np.array(decoded)

    logger.error(f"Encoder {type(encoder)} structure unknown. Returning raw codes.")
    return codes.astype(str)


def analyze_mistakes_simple(
    group_ids: np.ndarray,
    probs: np.ndarray,
    y_true: np.ndarray, # Group Truth
    y_pred: np.ndarray, # Group Pred
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
        # FIX: Flatten inputs to prevent ValueError: 2
        group_ids = np.array(group_ids).flatten()
        probs = np.array(probs).flatten()
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # 1. Create a Lookup for Group-Level Scores
        group_preds = pd.DataFrame({
            'global_group_id': group_ids,
            'Group_Prob': probs,
            'Group_True': y_true, 
            'Group_Pred': y_pred
        })

        # 2. Merge Scores back to PREPARED Transactions
        # original_df must have 'global_group_id'
        full_df = original_df.merge(group_preds, on='global_group_id', how='left')

        # 3. Handle Unscored/Unclustered Rows
        full_df['Group_Prob'] = full_df['Group_Prob'].fillna(0.0)
        full_df['Group_Pred'] = full_df['Group_Pred'].fillna(0).astype(int)
        
        # 4. Calculate TRUE Transaction-Level Metrics
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

        # 5. Define Mistake Types
        conditions = [
            (txn_y_true == 1) & (txn_y_pred == 1), # TP
            (txn_y_true == 0) & (txn_y_pred == 0), # TN
            (txn_y_true == 0) & (txn_y_pred == 1), # FP
            (txn_y_true == 1) & (txn_y_pred == 0), # FN
        ]
        choices = ['TP', 'TN', 'FP', 'FN']
        full_df['Result_Type'] = np.select(conditions, choices, default='Error')

        # 6. Save to CSV
        filename = f"mistake_analysis_{set_name}.csv"
        logger.info(f"Saving full transaction-level analysis to {filename}...")
        full_df.to_csv(filename, index=False)

        # 7. Log Top Group-Level Mistakes (for readability)
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

    # Fit on Train
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
        
        # 1. Transform raw data to prepped data (adds 'global_group_id')
        df_prepped = prep_dataset(df_raw)
        
        if len(df_prepped) == 0:
            logger.warning("Empty dataset, skipping.")
            return

        inference_ds = train_ds.from_dataset(
            train_ds, df_prepped, predict=True, stop_randomization=True
        )
        loader = inference_ds.to_dataloader(train=False, batch_size=256, num_workers=4)

        # Predict with return_x=True to access encoded groups
        raw_output = tft.predict(loader, mode="raw", return_x=True, return_index=False)
        
        raw_preds = raw_output[0]
        x = raw_output[1]

        probs = torch.softmax(raw_preds['prediction'], dim=-1)[:, 0, 1].cpu().numpy()
        y_true_group = x['decoder_target'][:, 0].cpu().numpy()
        y_pred_group = (probs > 0.5).astype(int)
        
        # 2. Extract and Decode Group IDs
        if 'groups' in x:
            encoded_groups = x['groups'][:, 0].cpu()
            group_encoder = train_ds.categorical_encoders['global_group_id']
            decoded_ids = robust_decode_groups(group_encoder, encoded_groups)
            
            # 3. Analyze on PREPPED DF (contains 'global_group_id')
            analyze_mistakes_simple(
                decoded_ids, 
                probs, 
                y_true_group, 
                y_pred_group, 
                df_prepped,  # <--- FIXED: Passing prepped data
                label_col=rc.field_config.label,
                set_name=set_name
            )
        else:
            logger.warning("Could not find 'groups' in x. Skipping analysis.")

    run_report("VAL SET", val_df)
    run_report("TEST SET", test_df)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/tft/eval_cluster_tft2.py <path_to_model.pt>")
        sys.exit(1)

    evaluate_checkpoint(sys.argv[1])