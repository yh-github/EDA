import numpy as np
import joblib
from sklearn.metrics import classification_report
from lex.data_loader import load_data, preprocess_data
from recurring_detector import RecurringDetector
from analyzer import calculate_f1


def evaluate_test_set():
    print("Loading data...")
    df = load_data()
    df = preprocess_data(df)

    # Replicate Split Logic exactly
    account_ids = df['accountId'].unique()
    np.random.seed(42)
    np.random.shuffle(account_ids)

    train_size = int(0.7 * len(account_ids))
    val_size = int(0.15 * len(account_ids))

    # Isolate Test Set
    test_accounts = account_ids[train_size + val_size:]
    test_df = df[df['accountId'].isin(test_accounts)].copy()

    print(f"Test accounts: {len(test_accounts)}")
    print(f"Test transactions: {len(test_df)}")

    # Load Model & Threshold
    print("Loading model...")
    try:
        clf = joblib.load('recurring_xgb_model.joblib')
        prob_threshold = joblib.load('xgb_threshold.joblib')
        print(f"Loaded threshold: {prob_threshold}")
    except:
        print("Model not found! Please run train_xgb.py first.")
        return

    # Generate Candidates on Test
    print("Generating candidates from Test set...")
    # Note: We must use the same detector params as training
    detector = RecurringDetector(
        interval_tolerance=40,
        min_transactions=2,
        amount_cv_threshold=1.0,
        dom_std_threshold=10.0,
        eps=0.29
    )

    test_candidates = detector.detect(test_df, return_candidates=True)

    if test_candidates.empty:
        print("No candidates found in test set.")
        return

    # Features
    feature_cols = [
        'interval_std', 'interval_median',
        'amount_cv', 'amount_std',
        'dom_std', 'dow_std',
        'count', 'days_span',
        'description_length', 'unique_descriptions'
    ]

    X_test = test_candidates[feature_cols].fillna(0)

    # Predict
    print("Predicting...")
    probs = clf.predict_proba(X_test)[:, 1]
    test_candidates['probability'] = probs

    # Filter by threshold
    selected = test_candidates[test_candidates['probability'] >= prob_threshold]

    # Map back to transactions
    print("Mapping results...")
    test_df['recurring_group_id'] = -1

    group_id_counter = 0
    for _, row in selected.iterrows():
        indices = row['indices']
        test_df.loc[indices, 'recurring_group_id'] = group_id_counter
        group_id_counter += 1

    # Calculate Metrics
    print("\n--- Test Set Results ---")
    f1 = calculate_f1(test_df)
    print(f"Test F1 Score: {f1:.4f}")

    # Detailed Report
    if 'isRecurring' in test_df.columns:
        y_true = test_df['isRecurring'].fillna(False).astype(bool)
        y_pred = test_df['recurring_group_id'] != -1

        # Calculate PR AUC
        # We need per-transaction probabilities for PR AUC.
        # For transactions in a candidate group, use the group's probability.
        # For others, probability is 0.

        # Create a map of group_id -> probability
        group_probs = dict(zip(range(group_id_counter), selected['probability']))

        # Map to transactions (default 0.0)
        y_scores = test_df['recurring_group_id'].map(group_probs).fillna(0.0)

        from sklearn.metrics import average_precision_score
        pr_auc = average_precision_score(y_true, y_scores)
        print(f"Test PR AUC: {pr_auc:.4f}")

        print("\nClassification Report (Per Transaction):")
        print(classification_report(y_true, y_pred))

        # Save results
        test_df.to_csv('test_results.csv', index=False)
        print("Saved test_results.csv")


if __name__ == "__main__":
    evaluate_test_set()
