import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Imports
from common.config import FieldConfig, FilterConfig, EmbModel
from common.embedder import EmbeddingService
from group_analyzer import RecurringGroupAnalyzer, GroupStabilityStatus
from common.log_utils import setup_logging

# Setup
setup_logging(Path('logs/'), "eval_clustering")
logger = logging.getLogger(__name__)

field_config = FieldConfig()
filter_config = FilterConfig()

# 1. Load Data (Full Dataset)
logger.info("Loading full dataset...")
full_df = pd.read_csv("data/rec_data2.csv")
full_df = full_df.dropna(subset=[field_config.date, field_config.amount, field_config.text])

# 2. Initialize Services
emb_service = EmbeddingService(
    model_name=EmbModel.MPNET,
    max_length=64,
    batch_size=256
)
# NOTE: Ensure your analyzer is tuned (eps, min_samples) in config.py
analyzer = RecurringGroupAnalyzer(filter_config, field_config)

unique_accounts = full_df[field_config.accountId].unique()
logger.info(f"Evaluating on {len(unique_accounts)} accounts...")

all_y_true = []
all_y_pred = []

# 3. Iterate Over Accounts
for account_id in tqdm(unique_accounts):
    # A. Get ALL transactions (Recurring AND Non-Recurring)
    account_df = full_df[full_df[field_config.accountId] == account_id].copy()

    if account_df.empty:
        continue

    # B. Embed EVERYTHING
    # (This tests the clusterer's ability to ignore noise)
    embeddings = emb_service.embed(account_df[field_config.text].tolist())

    # C. Run Unsupervised Analysis
    found_groups = analyzer.analyze_account(account_df, embeddings)

    # D. Generate Predictions based on Stability
    # Default prediction is 0 (Not Recurring)
    account_df['pred_label'] = 0

    for grp in found_groups:
        # The Core Logic: Only STABLE groups are predicted as Recurring (1)
        if grp.status == GroupStabilityStatus.STABLE:
            # Mark the transactions in this group as 1
            mask = account_df[field_config.trId].isin(grp.transaction_ids)
            account_df.loc[mask, 'pred_label'] = 1

    # E. Collect Results
    all_y_true.extend(account_df[field_config.label].tolist())
    all_y_pred.extend(account_df['pred_label'].tolist())

# 4. Final Evaluation
print("\n" + "=" * 40)
print("CLUSTER-BASED CLASSIFIER RESULTS")
print("=" * 40)
print(classification_report(all_y_true, all_y_pred, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(all_y_true, all_y_pred))