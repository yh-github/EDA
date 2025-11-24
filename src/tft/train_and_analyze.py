import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet

from common.config import FieldConfig, EmbModel, ExperimentConfig
from common.data import create_train_val_test_split
from common.embedder import EmbeddingService
from common.feature_processor import FeatProcParams
from common.log_utils import setup_logging
from tft.tft_data_clustered import prepare_clustered_tft_data, build_clustered_tft_dataset
from tft.tft_runner import TFTRunner

# Setup Logging
setup_logging(Path("logs/"), "analysis_run")
logger = logging.getLogger("analysis")

# --- Configuration ---
HYPER_PARAMS = {
    "learning_rate": 0.0001,
    "hidden_size": 128,
    "attention_head_size": 2,
    "dropout": 0.2,
    "hidden_continuous_size": 128,
    "output_size": 2,
    "gradient_clip_val": 0.84
}
MAX_ENCODER_LENGTH = 64


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
        use_categorical_amount=False
    )

    # Prepare Clustered Data
    train_prepped, pca_model, processor, meta = prepare_clustered_tft_data(
        train_df, field_config, feat_params, emb_service, fit_processor=True
    )
    val_prepped, _, _, _ = prepare_clustered_tft_data(
        val_df, field_config, embedding_service=emb_service,
        pca_model=pca_model, processor=processor, fit_processor=False
    )

    # Build Datasets
    train_ds = build_clustered_tft_dataset(
        train_prepped, field_config, meta, max_encoder_length=MAX_ENCODER_LENGTH
    )
    val_ds = TimeSeriesDataSet.from_dataset(train_ds, val_prepped, predict=True, stop_randomization=True)

    train_loader = train_ds.to_dataloader(train=True, batch_size=128, num_workers=4)
    val_loader = val_ds.to_dataloader(train=False, batch_size=128, num_workers=4)

    # 2. Train Model via Runner
    # We do not use pos_weight here to stay consistent with previous analysis script,
    # but we benefit from the ManualMetricCallback logging.
    runner = TFTRunner(train_ds, train_loader, val_loader, max_epochs=3)

    logger.info("Training TFT...")
    trainer, tft = runner.train_single(HYPER_PARAMS)

    # 3. Generate Interpretability (Feature Importance)
    logger.info("Analyzing Feature Importance...")

    prediction_output = tft.predict(val_loader, mode="raw", return_x=True)

    if isinstance(prediction_output, tuple) and len(prediction_output) == 3:
        raw_predictions, x, _ = prediction_output
    elif isinstance(prediction_output, tuple) and len(prediction_output) == 2:
        raw_predictions, x = prediction_output
    else:
        raise ValueError(f"Unexpected output from tft.predict: {type(prediction_output)}")

    interpretation = tft.interpret_output(raw_predictions, reduction="sum")

    def print_importance(name, importance_tensor, feature_names):
        imp = importance_tensor.cpu().numpy()
        imp = imp / imp.sum() * 100
        indices = np.argsort(imp)[::-1]

        print(f"\n--- {name} Feature Importance ---")
        for idx in indices:
            feat_name = feature_names[idx]
            score = imp[idx]
            if score < 0.1: continue
            print(f"{feat_name:<30} : {score:.1f}%")

    print_importance("Static", interpretation["static_variables"], tft.static_variables)
    print_importance("Encoder (History)", interpretation["encoder_variables"], tft.encoder_variables)
    print_importance("Decoder (Target)", interpretation["decoder_variables"], tft.decoder_variables)

    # 4. Analyze Worst Accounts
    logger.info("Identifying Worst Performing Accounts...")

    logits = raw_predictions.prediction
    probs = torch.softmax(logits, dim=-1)

    results_data = []
    group_id_encoder = tft.dataset.categorical_encoders['global_group_id']
    batch_group_codes = x['decoder_cat'][:, 0, 0].cpu()
    decoded_groups = group_id_encoder.inverse_transform(batch_group_codes)

    y_true = x['decoder_target'][:, 0].cpu().numpy()
    y_prob_rec = probs[:, 0, 1].cpu().numpy()

    sample_losses = []
    for t, p in zip(y_true, y_prob_rec):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        loss = - (t * np.log(p) + (1 - t) * np.log(1 - p))
        sample_losses.append(loss)

    pred_df = pd.DataFrame({
        "global_group_id": decoded_groups,
        "prob_recurring": y_prob_rec,
        "true_label": y_true,
        "loss": sample_losses
    })

    def extract_account(grp_id):
        return grp_id.split("_amt_")[0]

    pred_df['accountId'] = pred_df['global_group_id'].apply(extract_account)

    account_stats = pred_df.groupby('accountId').agg(
        avg_loss=('loss', 'mean'),
        count=('loss', 'count')
    ).sort_values('avg_loss', ascending=False)

    worst_accounts = account_stats.head(3).index.tolist()
    logger.info(f"Worst 3 Accounts by Loss: {worst_accounts}")

    print("\n" + "=" * 50)
    print("DEBUGGING WORST ACCOUNTS")
    print("=" * 50)

    for acc_id in worst_accounts:
        print(f"\n>>> Account: {acc_id} (Avg Loss: {account_stats.loc[acc_id, 'avg_loss']:.4f})")
        acc_preds = pred_df[pred_df['accountId'] == acc_id]
        acc_raw_df = val_prepped[val_prepped['accountId'] == acc_id].copy()

        display_rows = []
        for _, row in acc_preds.iterrows():
            grp_id = row['global_group_id']
            grp_data = acc_raw_df[acc_raw_df['global_group_id'] == grp_id].sort_values(field_config.date)
            if grp_data.empty: continue

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
        if not display_df.empty:
            display_df = display_df.sort_values("Loss", ascending=False)
            print(display_df.to_string(index=False))
        else:
            print("  (No matching validation samples found)")


if __name__ == "__main__":
    train_and_analyze()