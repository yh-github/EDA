import torch
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# Project imports
from config import FieldConfig
from tft_data import build_tft_dataset
from data import create_train_val_test_split

from log_utils import setup_logging

setup_logging(Path("logs/"), "eval_tft")
logger = logging.getLogger(__name__)

# Path to your best checkpoint (Optuna saves these in `lightning_logs` or you can save manually)
# UPDATE THIS AFTER TUNING
CHECKPOINT_PATH = "lightning_logs/version_0/checkpoints/epoch=...ckpt"


def evaluate():
    # 1. Load Data
    field_config = FieldConfig()
    full_df = pd.read_csv("data/rec_data2.csv").dropna()

    # We need the TEST set this time
    _, _, test_df = create_train_val_test_split(test_size=0.2, val_size=0.2, full_df=full_df, random_state=42)

    # 2. Load Model
    # Note: TFT stores the dataset parameters inside the checkpoint!
    # We don't need to rebuild the definition manually.
    if not Path(CHECKPOINT_PATH).exists():
        logger.error(f"Checkpoint not found at {CHECKPOINT_PATH}")
        return

    tft = TemporalFusionTransformer.load_from_checkpoint(CHECKPOINT_PATH)
    logger.info("Model loaded successfully.")

    # 3. Prepare Test Loader
    # We create a dataset from the model's internal dataset parameters
    test_ds = TimeSeriesDataSet.from_dataset(tft.dataset, test_df, predict=True, stop_randomization=True)
    test_loader = test_ds.to_dataloader(train=False, batch_size=128, num_workers=4)

    # 4. Predict
    # return_y=True gives us the actual targets to compare against
    logger.info("Running predictions...")
    raw_predictions, x = tft.predict(test_loader, mode="raw", return_x=True)

    # raw_predictions.output is (Batch, Prediction_Length, Classes) -> (B, 1, 2)
    # We want the probability of Class 1 (Recurring)
    y_pred_proba = raw_predictions.prediction[:, 0, 1].cpu().numpy()
    y_pred_label = (y_pred_proba > 0.5).astype(int)

    # Get True Labels (also formatted by the loader)
    # x["decoder_target"] contains the true labels
    y_true = x["decoder_target"][:, 0].cpu().numpy()

    # 5. Metrics
    logger.info("\n" + "=" * 40)
    logger.info("TFT CLASSIFICATION REPORT")
    logger.info("=" * 40)
    print(classification_report(y_true, y_pred_label, digits=4))

    auc = roc_auc_score(y_true, y_pred_proba)
    logger.info(f"ROC-AUC: {auc:.4f}")

    # 6. Interpretability (The "Top Notch" part)
    logger.info("Generating Interpretability Plots...")

    # Variable Importance
    interpretation = tft.interpret_output(raw_predictions, reduction="sum")
    tft.plot_interpretation(interpretation)
    plt.savefig("logs/variable_importance.png")
    logger.info("Saved logs/variable_importance.png")

    # Attention (Sample)
    # Plot prediction for a few random samples to see Attention weights
    # This shows 'where' in history the model looked
    for i in range(3):
        tft.plot_prediction(x, raw_predictions, idx=i, add_loss_to_title=True)
        plt.savefig(f"logs/prediction_sample_{i}.png")
        plt.close()
    logger.info("Saved prediction sample plots.")


if __name__ == "__main__":
    evaluate()
