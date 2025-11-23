import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# --- Imports ---
from common.config import FieldConfig, FilterConfig, EmbModel
from common.embedder import EmbeddingService
from greedy_analyzer import GreedyGroupAnalyzer, GroupStabilityStatus
from common.log_utils import setup_logging
import logging

# --- Configuration ---
# Set to True to use your trained HybridModel (Stage 3)
# Set to False to use raw MPNET text embeddings (Stage 2)
USE_TRAINED_MODEL = False
MODEL_CHECKPOINT_PATH = "cache/results/best_model_dir"  # Update if USE_TRAINED_MODEL is True

# Setup Logging
setup_logging(Path('logs/'), "eval_greedy")
logger = logging.getLogger(__name__)


def main():
    # 1. Load Data
    logger.info("Loading full dataset...")
    field_config = FieldConfig()
    filter_config = FilterConfig()

    try:
        full_df = pd.read_csv("data/rec_data2.csv")
        full_df = full_df.dropna(
            subset=[field_config.date, field_config.amount, field_config.text]
        )
    except FileNotFoundError:
        logger.error("Data file not found.")
        return

    # 2. Initialize Embedder / Model
    if USE_TRAINED_MODEL:
        logger.info("Loading trained HybridModel for 'Smart Embeddings'...")
        # Note: You need to implement loading logic based on how you saved the model
        # This is a placeholder for the concept we discussed
        # model = HybridModel.load(MODEL_CHECKPOINT_PATH)
        # model.eval()
        pass
    else:
        logger.info("Initializing standard Text Embedder (MPNET)...")
        emb_service = EmbeddingService(
            model_name=EmbModel.MPNET,
            max_length=64,
            batch_size=256
        )

    # 3. Initialize the New Analyzer
    analyzer = GreedyGroupAnalyzer(filter_config, field_config)

    # Adjust thresholds if necessary (optional, defaults are in the class)
    # analyzer.sim_threshold = 0.92
    # analyzer.amount_tol_abs = 1.50

    unique_accounts = full_df[field_config.accountId].unique()
    logger.info(f"Evaluating on {len(unique_accounts)} accounts...")

    all_y_true = []
    all_y_pred = []

    found_patterns_count = 0

    # 4. Iterate Over Accounts
    for account_id in tqdm(unique_accounts):
        # A. Get ALL transactions for this account
        account_df = full_df[full_df[field_config.accountId] == account_id].copy()

        if account_df.empty:
            continue

        # B. Generate Embeddings
        # This is the interface point: passed as numpy array [N, D]
        if USE_TRAINED_MODEL:
            # TODO: Implement batch conversion and model forward pass
            # embeddings = model.get_representation(...).cpu().numpy()
            embeddings = np.zeros((len(account_df), 128))  # Placeholder
        else:
            embeddings = emb_service.embed(account_df[field_config.text].tolist())

        # C. Run Greedy Analysis
        found_groups = analyzer.analyze_account(account_df, embeddings)

        # D. Generate Predictions (Unsupervised -> Supervised Label)
        # Default prediction is 0 (Not Recurring)
        account_df['pred_label'] = 0

        for grp in found_groups:
            if grp.status == GroupStabilityStatus.STABLE:
                found_patterns_count += 1
                # Mark these specific transaction IDs as recurring (1)
                mask = account_df[field_config.trId].isin(grp.transaction_ids)
                account_df.loc[mask, 'pred_label'] = 1

        # E. Collect Results for Classification Report
        all_y_true.extend(account_df[field_config.label].tolist())
        all_y_pred.extend(account_df['pred_label'].tolist())

    # 5. Final Evaluation
    logger.info("\n" + "=" * 40)
    logger.info(f"GREEDY ANALYZER RESULTS (Found {found_patterns_count} patterns)")
    logger.info("=" * 40)

    # Classification Report (Precision/Recall/F1)
    report = classification_report(all_y_true, all_y_pred, digits=4)
    logger.info("\n" + report)

    # Confusion Matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")


if __name__ == "__main__":
    main()