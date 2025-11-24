import logging
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import CrossEntropy

from common.config import FieldConfig, EmbModel, ExperimentConfig
from common.data import create_train_val_test_split
from common.embedder import EmbeddingService
from common.feature_processor import FeatProcParams
from common.log_utils import setup_logging
from tft.tft_data_clustered import prepare_clustered_tft_data, build_clustered_tft_dataset

# Setup Logging
setup_logging(Path("logs/"), "analysis_run")
logger = logging.getLogger("analysis")

# --- Configuration ---
# "Good" Hyperparameters (based on typical convergence for this data scale)
HYPER_PARAMS = {
    "learning_rate": 0.0001,
    "hidden_size": 128,
    "attention_head_size": 2,
    "dropout": 0.2,
    "hidden_continuous_size": 128,
    "output_size": 2,  # Binary
    "max_encoder_length": 64
}


def train_and_analyze():
    # 1. Load & Prepare Data
    exp_config = ExperimentConfig()

    logger.info("Loading and Preparing Data...")
    field_config = FieldConfig()
    full_df = pd.read_csv("data/rec_data2.csv").dropna(
        subset=[field_config.date, field_config.amount, field_config.text]
    )

    # Split by Account to ensure fair "Unseen" testing
    train_df, val_df, _ = create_train_val_test_split(
        test_size=0.2, val_size=0.2, full_df=full_df, random_state=exp_config.random_state
    )

    # Initialize Services
    emb_service = EmbeddingService(model_name=EmbModel.MPNET, max_length=64, batch_size=512)
    feat_params = FeatProcParams(
        use_cyclical_dates=True,
        use_continuous_amount=True,
        use_categorical_amount=False  # Clustered approach handles amount grouping natively
    )

    # Prepare Clustered Data (Grouping by Amount Similarity)
    # We fit the processor on Train and apply to Val
    train_prepped, pca_model, processor, meta = prepare_clustered_tft_data(
        train_df, field_config, feat_params, emb_service, fit_processor=True
    )
    val_prepped, _, _, _ = prepare_clustered_tft_data(
        val_df, field_config, embedding_service=emb_service,
        pca_model=pca_model, processor=processor, fit_processor=False
    )

    # Build Datasets
    train_ds = build_clustered_tft_dataset(
        train_prepped, field_config, meta, max_encoder_length=HYPER_PARAMS["max_encoder_length"]
    )
    val_ds = TimeSeriesDataSet.from_dataset(train_ds, val_prepped, predict=True, stop_randomization=True)

    train_loader = train_ds.to_dataloader(train=True, batch_size=128, num_workers=4)
    val_loader = val_ds.to_dataloader(train=False, batch_size=128, num_workers=4)

    # 2. Train Model
    logger.info("Training TFT...")
    tft = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=HYPER_PARAMS["learning_rate"],
        hidden_size=HYPER_PARAMS["hidden_size"],
        attention_head_size=HYPER_PARAMS["attention_head_size"],
        dropout=HYPER_PARAMS["dropout"],
        hidden_continuous_size=HYPER_PARAMS["hidden_continuous_size"],
        output_size=HYPER_PARAMS["output_size"],
        loss=CrossEntropy(),
        log_interval=10
    )

    trainer = pl.Trainer(
        max_epochs=10,  # Enough for analysis; increase for production
        accelerator="auto",
        gradient_clip_val=0.84, # TODO: extern into a parameter
        enable_checkpointing=False,
        logger=False
    )
    trainer.fit(tft, train_loader, val_dataloaders=val_loader)

    # 3. Generate Interpretability (Feature Importance)
    logger.info("Analyzing Feature Importance...")

    # interpret_output returns a dict with 'attention' and 'static_variables', 'encoder_variables', etc.
    raw_predictions, x = tft.predict(val_loader, mode="raw", return_x=True)
    interpretation = tft.interpret_output(raw_predictions, reduction="sum")

    # Helper to normalize and print importance
    def print_importance(name, importance_tensor, feature_names):
        # Normalize to sum=100%
        imp = importance_tensor.cpu().numpy()
        imp = imp / imp.sum() * 100

        # Sort
        indices = np.argsort(imp)[::-1]

        print(f"\n--- {name} Feature Importance ---")
        for idx in indices:
            feat_name = feature_names[idx]
            score = imp[idx]
            if score < 0.1: continue  # Skip noise
            print(f"{feat_name:<30} : {score:.1f}%")

    # A. Static Variables (Account ID, etc.)
    print_importance("Static", interpretation["static_variables"], tft.static_variables)

    # B. Encoder Variables (History)
    print_importance("Encoder (History)", interpretation["encoder_variables"], tft.encoder_variables)

    # C. Decoder Variables (Known Future / Target Transaction)
    print_importance("Decoder (Target)", interpretation["decoder_variables"], tft.decoder_variables)

    # 4. Analyze Worst Accounts
    logger.info("Identifying Worst Performing Accounts...")

    # Extract Probabilities
    # raw_predictions.prediction shape: [Batch, Prediction_Len, Classes]
    # We want the probability of the TRUE class.

    logits = raw_predictions.prediction
    probs = torch.softmax(logits, dim=-1)  # [Batch, 1, 2]

    # Extract metadata for mapping back
    # The loader might shuffle, but 'x' aligns with 'raw_predictions'
    # val_ds has a method to decode the categorical group_ids

    # We need the DataFrame corresponding to these predictions to get Account IDs
    # Since we used `predict(..., return_x=True)`, we can't easily map back to the original DF rows
    # *unless* we are careful. The safest way is to iterate the loader again with the model in eval mode
    # or assume order is preserved if shuffle=False.

    # Let's construct a results dataframe from the validation set directly
    # Note: This is an approximation. For perfect alignment, we'd iterate the dataloader manually.

    results_data = []

    # To map back, we need the 'global_group_id' which is encoded in x['decoder_cat']
    # 'global_group_id' is the group_id at index 0 in group_ids list
    group_id_encoder = tft.dataset.categorical_encoders['global_group_id']

    # Flatten batches
    batch_group_codes = x['decoder_cat'][:, 0, 0].cpu()  # Assuming global_group_id is first group
    decoded_groups = group_id_encoder.inverse_transform(batch_group_codes)

    # True Labels (from decoder_target)
    y_true = x['decoder_target'][:, 0].cpu().numpy()
    # Predicted Probabilities for Class 1 (Recurring)
    y_prob_rec = probs[:, 0, 1].cpu().numpy()

    # Calculate Loss per Sample (Log Loss / Cross Entropy)
    # High Loss = Bad Prediction
    sample_losses = []
    for t, p in zip(y_true, y_prob_rec):
        # Clip for stability
        p = np.clip(p, 1e-7, 1 - 1e-7)
        loss = - (t * np.log(p) + (1 - t) * np.log(1 - p))
        sample_losses.append(loss)

    # Build DataFrame of Predictions
    pred_df = pd.DataFrame({
        "global_group_id": decoded_groups,
        "prob_recurring": y_prob_rec,
        "true_label": y_true,
        "loss": sample_losses
    })

    # We need to map 'global_group_id' back to 'accountId'
    # global_group_id format: "{acc_id}_amt_{amt}"
    # Let's extract it assuming the format we created in tft_data_clustered.py
    # "f'{acc_id}_amt_{cluster_id}'"

    def extract_account(grp_id):
        # Robust split: split by "_amt_" and take the first part
        return grp_id.split("_amt_")[0]

    pred_df['accountId'] = pred_df['global_group_id'].apply(extract_account)

    # Aggregate Loss by Account
    # We use Mean Loss to find accounts that are *consistently* hard to predict
    account_stats = pred_df.groupby('accountId').agg(
        avg_loss=('loss', 'mean'),
        count=('loss', 'count')
    ).sort_values('avg_loss', ascending=False)

    worst_accounts = account_stats.head(3).index.tolist()

    logger.info(f"Worst 3 Accounts by Loss: {worst_accounts}")

    # 5. Output Detailed Data for Worst Accounts
    # We merge prediction info back with the detailed features from val_prepped

    print("\n" + "=" * 50)
    print("DEBUGGING WORST ACCOUNTS")
    print("=" * 50)

    for acc_id in worst_accounts:
        print(f"\n>>> Account: {acc_id} (Avg Loss: {account_stats.loc[acc_id, 'avg_loss']:.4f})")

        # Get all predictions for this account
        acc_preds = pred_df[pred_df['accountId'] == acc_id]

        # Get original details (Text, Date, Amount) from the prepped dataframe
        # We join on global_group_id. Note: val_prepped has multiple rows per group (history),
        # pred_df has 1 row per group (the prediction target).
        # We want to show the TARGET transaction details.

        # In TFT validation mode (predict=True), the dataset returns the last available window.
        # So we grab the last row of each group from val_prepped to show what was predicted.

        # Filter val_prepped for this account
        acc_raw_df = val_prepped[val_prepped['accountId'] == acc_id].copy()

        # We want to display: Date, Amount, Text, True, Prob
        # We can iterate the predictions and find the matching rows

        display_rows = []
        for _, row in acc_preds.iterrows():
            grp_id = row['global_group_id']

            # Get the specific group data
            grp_data = acc_raw_df[acc_raw_df['global_group_id'] == grp_id].sort_values(field_config.date)
            if grp_data.empty: continue

            # The target is the last element in the sequence
            target_row = grp_data.iloc[-1]

            display_rows.append({
                "Group ID": grp_id,
                "Date": target_row[field_config.date],
                "Amount": f"{target_row[field_config.amount]:.2f}",
                "Text": target_row[field_config.text],
                "True Label": int(row['true_label']),
                "Model Prob": f"{row['prob_recurring']:.4f}",
                "Loss": f"{row['loss']:.4f}"
            })

        display_df = pd.DataFrame(display_rows)
        # Sort by Loss (descending) to show the biggest errors first
        if not display_df.empty:
            display_df = display_df.sort_values("Loss", ascending=False)
            print(display_df.to_string(index=False))
        else:
            print("  (No matching validation samples found - likely all filtered by min_encoder_length)")


if __name__ == "__main__":
    train_and_analyze()