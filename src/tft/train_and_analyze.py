import logging
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet

from common.config import FieldConfig, EmbModel, ExperimentConfig
from common.data import create_train_val_test_split
from common.embedder import EmbeddingService
from common.feature_processor import FeatProcParams
from common.log_utils import setup_logging
from tft.tft_data_clustered import prepare_clustered_tft_data, build_clustered_tft_dataset
from tft.tft_runner import TFTRunner

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
# Define where to cache the model
MODEL_CACHE_PATH = "cache/tft_models/analysis_model_v1.pt"


def train_and_analyze():
    exp_config = ExperimentConfig()

    logger.info("Loading and Preparing Data...")
    field_config = FieldConfig()
    full_df = pd.read_csv("data/rec_data2.csv").dropna(
        subset=[field_config.date, field_config.amount, field_config.text]
    )

    train_df, val_df, _ = create_train_val_test_split(
        test_size=0.2, val_size=0.2, full_df=full_df, random_state=exp_config.random_state
    )

    emb_service = EmbeddingService(model_name=EmbModel.MPNET, max_length=64, batch_size=512)
    feat_params = FeatProcParams(
        use_cyclical_dates=True,
        use_continuous_amount=True,
        use_categorical_amount=False
    )

    # Prepare Data
    train_prepped, pca_model, processor, meta = prepare_clustered_tft_data(
        train_df, field_config, feat_params, emb_service, fit_processor=True
    )
    val_prepped, _, _, _ = prepare_clustered_tft_data(
        val_df, field_config, embedding_service=emb_service,
        pca_model=pca_model, processor=processor, fit_processor=False
    )

    train_ds = build_clustered_tft_dataset(
        train_prepped, field_config, meta, max_encoder_length=MAX_ENCODER_LENGTH
    )
    val_ds = TimeSeriesDataSet.from_dataset(train_ds, val_prepped, predict=True, stop_randomization=True)

    train_loader = train_ds.to_dataloader(train=True, batch_size=128, num_workers=4)
    val_loader = val_ds.to_dataloader(train=False, batch_size=128, num_workers=4)

    # --- Train or Load Model ---
    runner = TFTRunner(train_ds, train_loader, val_loader, max_epochs=3)

    logger.info(f"Training (or loading from {MODEL_CACHE_PATH})...")
    trainer, tft = runner.train_single(HYPER_PARAMS, model_path=MODEL_CACHE_PATH)

    # --- Analysis ---
    logger.info("Analyzing Feature Importance...")

    # Get predictions (TFT predict handles device automatically usually)
    # We use the trainer's data loader directly
    prediction_output = tft.predict(val_loader, mode="raw", return_x=True)

    if isinstance(prediction_output, tuple) and len(prediction_output) == 3:
        raw_predictions, x, _ = prediction_output
    elif isinstance(prediction_output, tuple) and len(prediction_output) == 2:
        raw_predictions, x = prediction_output
    else:
        raise ValueError("Unexpected output format from tft.predict")

    # Note: Feature importance calculation is slow on full dataset
    # If caching model, this might still be the bottleneck.
    interpretation = tft.interpret_output(raw_predictions, reduction="sum")

    def print_importance(name, importance_tensor, feature_names):
        imp = importance_tensor.cpu().numpy()
        imp = imp / imp.sum() * 100
        indices = np.argsort(imp)[::-1]
        print(f"\n--- {name} Feature Importance ---")
        for idx in indices:
            if imp[idx] < 0.1: continue
            print(f"{feature_names[idx]:<30} : {imp[idx]:.1f}%")

    print_importance("Static", interpretation["static_variables"], tft.static_variables)
    print_importance("Encoder", interpretation["encoder_variables"], tft.encoder_variables)
    print_importance("Decoder", interpretation["decoder_variables"], tft.decoder_variables)

    # --- Worst Accounts ---
    logger.info("Identifying Worst Performing Accounts...")
    probs = torch.softmax(raw_predictions.prediction, dim=-1)

    y_true = x['decoder_target'][:, 0].cpu().numpy()
    y_prob_rec = probs[:, 0, 1].cpu().numpy()

    # Reconstruct Group IDs
    group_id_encoder = tft.dataset.categorical_encoders['global_group_id']
    batch_group_codes = x['decoder_cat'][:, 0, 0].cpu()
    decoded_groups = group_id_encoder.inverse_transform(batch_group_codes)

    # Calculate Losses
    sample_losses = []
    for t, p in zip(y_true, y_prob_rec):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        sample_losses.append(- (t * np.log(p) + (1 - t) * np.log(1 - p)))

    pred_df = pd.DataFrame({
        "global_group_id": decoded_groups,
        "prob_recurring": y_prob_rec,
        "true_label": y_true,
        "loss": sample_losses
    })

    pred_df['accountId'] = pred_df['global_group_id'].apply(lambda g: g.split("_amt_")[0])

    account_stats = pred_df.groupby('accountId').agg(
        avg_loss=('loss', 'mean'),
        count=('loss', 'count')
    ).sort_values('avg_loss', ascending=False)

    worst_accounts = account_stats.head(3).index.tolist()
    logger.info(f"Worst 3 Accounts: {worst_accounts}")

    print("\n" + "=" * 50)
    for acc_id in worst_accounts:
        print(f"\n>>> Account: {acc_id} (Avg Loss: {account_stats.loc[acc_id, 'avg_loss']:.4f})")
        acc_preds = pred_df[pred_df['accountId'] == acc_id]
        acc_raw = val_prepped[val_prepped['accountId'] == acc_id].copy()

        # Optim: Sort once
        acc_raw = acc_raw.sort_values(['global_group_id', field_config.date])
        raw_groups = acc_raw.groupby('global_group_id')

        rows = []
        for _, row in acc_preds.iterrows():
            gid = row['global_group_id']
            if gid in raw_groups.groups:
                target = raw_groups.get_group(gid).iloc[-1]
                rows.append({
                    "Group": gid,
                    "Date": target[field_config.date],
                    "Amt": f"{target[field_config.amount]:.2f}",
                    "Text": target[field_config.text],
                    "True": int(row['true_label']),
                    "Prob": f"{row['prob_recurring']:.2f}",
                    "Loss": f"{row['loss']:.2f}"
                })

        if rows:
            print(pd.DataFrame(rows).sort_values("Loss", ascending=False).to_string(index=False))


if __name__ == "__main__":
    train_and_analyze()