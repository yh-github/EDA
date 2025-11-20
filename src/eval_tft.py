import torch
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from config import FieldConfig
from tft_data import prepare_tft_data
from data import create_train_val_test_split

from log_utils import setup_logging

setup_logging(Path("logs/"), "eval_tft")
logger = logging.getLogger(__name__)

# UPDATE THIS AFTER TUNING
CHECKPOINT_PATH = "lightning_logs/version_0/checkpoints/epoch=...ckpt"


def evaluate():
    # 1. Load Data
    field_config = FieldConfig()
    full_df = pd.read_csv("data/rec_data2.csv").dropna()

    # We need the TEST set this time
    _, _, test_df = create_train_val_test_split(test_size=0.2, val_size=0.2, full_df=full_df, random_state=42)

    # Run the preparation pipeline to add 'time_idx' and cast types
    logger.info("Preparing test data...")
    test_df_prepped = prepare_tft_data(test_df, field_config)

    # 2. Load Model
    if not Path(CHECKPOINT_PATH).exists():
        logger.error(f"Checkpoint not found at {CHECKPOINT_PATH}")
        return

    tft = TemporalFusionTransformer.load_from_checkpoint(CHECKPOINT_PATH)
    logger.info("Model loaded successfully.")

    # 3. Prepare Test Loader
    # Use the prepped dataframe
    test_ds = TimeSeriesDataSet.from_dataset(tft.dataset, test_df_prepped, predict=True, stop_randomization=True)
    test_loader = test_ds.to_dataloader(train=False, batch_size=128, num_workers=4)

    # 4. Predict
    logger.info("Running predictions...")
    # mode="raw" returns a dictionary containing 'prediction' (logits)
    raw_predictions, x = tft.predict(test_loader, mode="raw", return_x=True)

    # Convert Logits to Probabilities
    # raw_predictions.prediction shape: (Batch, Prediction_Length=1, Classes=2)
    logits = raw_predictions.prediction
    probabilities = torch.softmax(logits, dim=-1)

    # Extract probability of Class 1 (Recurring)
    y_pred_proba = probabilities[:, 0, 1].cpu().numpy()

    # Now we can safely use threshold
    y_pred_label = (y_pred_proba > 0.5).astype(int)

    # Get True Labels
    y_true = x["decoder_target"][:, 0].cpu().numpy()

    # 5. Metrics
    logger.info("\n" + "=" * 40)
    logger.info("TFT CLASSIFICATION REPORT")
    logger.info("=" * 40)
    print(classification_report(y_true, y_pred_label, digits=4))

    # Handle case where y_true only has 1 class (e.g. small test set)
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
        logger.info(f"ROC-AUC: {auc:.4f}")
    except ValueError:
        logger.warning("ROC-AUC could not be calculated (only one class present in y_true).")

    # 6. Interpretability
    logger.info("Generating Interpretability Plots...")

    # Variable Importance
    interpretation = tft.interpret_output(raw_predictions, reduction="sum")
    tft.plot_interpretation(interpretation)
    plt.savefig("logs/variable_importance.png")
    logger.info("Saved logs/variable_importance.png")

    # Attention (Sample)
    # We pick valid indices from the test set
    for i in range(min(3, len(test_df_prepped))):
        tft.plot_prediction(x, raw_predictions, idx=i, add_loss_to_title=True)
        plt.savefig(f"logs/prediction_sample_{i}.png")
        plt.close()
    logger.info("Saved prediction sample plots.")


if __name__ == "__main__":
    evaluate()