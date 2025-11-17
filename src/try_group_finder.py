import pandas as pd
from pathlib import Path

# --- Your existing classes ---
from config import FieldConfig, FilterConfig, EmbModel
from embedder import EmbeddingService
from log_utils import setup_logging

# --- Our new classes ---
from group_analyzer import RecurringGroupAnalyzer, GroupStabilityStatus

# --- 1. Setup (Logging, Configs) ---
setup_logging(Path('logs/'), "analyzer_test")

field_config = FieldConfig()
filter_config = FilterConfig()  # Uses defaults from config.py

# --- 2. Load Data ---
# This would be your full dataset
try:
    full_df = pd.read_csv("data/rec_data2.csv")
    full_df = full_df.dropna(
        subset=[
            field_config.date,
            field_config.amount,
            field_config.text,
            field_config.label
        ]
    )
except FileNotFoundError:
    print("Error: data/rec_data2.csv not found.")
    exit()

# --- 3. Get Inputs for a Single Account ---

# For this example, we'll pick an account to test
# In a real system, you'd loop over all account IDs
TEST_ACCOUNT_ID = full_df[field_config.accountId].iloc[0]

# a) Get all transactions for this account
account_df = full_df[
    full_df[field_config.accountId] == TEST_ACCOUNT_ID
    ].copy()

# b) Filter to ONLY recurring transactions.
# Here, we use the TRUE label. In production, you would use
# your Stage 1 model's *predicted* label.
recurring_df = account_df[account_df[field_config.label] == 1].copy()

if recurring_df.empty:
    print(f"Account {TEST_ACCOUNT_ID} has no recurring transactions to analyze.")
else:
    # c) Get text embeddings for these transactions
    # (This uses your project's caching system!)
    print(f"Loading embedder...")
    emb_service = EmbeddingService(
        model_name=EmbModel.MiniLM_L12,  # Or any model you prefer
        max_length=64,
        batch_size=256
    )

    print(f"Getting embeddings for {len(recurring_df)} transactions...")
    embeddings = emb_service.embed(recurring_df[field_config.text].tolist())

    # --- 4. Run Analysis ---
    print(f"Analyzing account {TEST_ACCOUNT_ID}...")
    analyzer = RecurringGroupAnalyzer(
        filter_config=filter_config,
        field_config=field_config
    )

    analysis_results = analyzer.analyze_account(recurring_df, embeddings)

    # --- 5. Show Results ---
    print("\n" + "=" * 30)
    print(f"ANALYSIS COMPLETE FOR ACCOUNT: {TEST_ACCOUNT_ID}")
    print(f"Found {len(analysis_results)} potential recurring groups.")
    print("=" * 30)

    for result in analysis_results:
        print(f"\n--- Cluster {result.cluster_id} ({result.status.name}) ---")
        print(f"  Transactions: {len(result.transaction_ids)}")

        if result.status == GroupStabilityStatus.STABLE:
            print(f"  Forecast:")
            print(f"    Next Date: {result.next_predicted_date.strftime('%Y-%m-%d')}")
            print(f"    Amount: {result.predicted_amount:,.2f}")
            print(f"    Cycle: {result.cycle_days:.1f} days")
        elif result.status == GroupStabilityStatus.NOT_STABLE_AMOUNT_VARIANCE:
            print(f"  Analysis Failed: Amount stdev ({result.amount_std:.2f}) "
                  f"> threshold ({filter_config.amount_std_threshold:.2f})")
        elif result.status == GroupStabilityStatus.NOT_STABLE_DATE_VARIANCE:
            print(f"  Analysis Failed: Date delta stdev ({result.date_delta_std_days:.1f}) "
                  f"> threshold ({filter_config.date_std_threshold:.1f})")
        else:
            print(f"  Analysis Failed: {result.status.name}")