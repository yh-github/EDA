import joblib
from sklearn.metrics import classification_report, average_precision_score
from lex.data_loader import load_lex_splits
from lex.recurring_detector import RecurringDetector
from lex.analyzer import calculate_f1
from common.config import FieldConfig


def evaluate_test_set():
    print("Loading Test split...")
    field_config = FieldConfig()
    # We ignore train/val here
    _, _, test_df = load_lex_splits()

    print(f"Test transactions: {len(test_df)}")

    # Load Model
    try:
        clf = joblib.load('recurring_xgb_model.joblib')
        prob_threshold = joblib.load('xgb_threshold.joblib')
        print(f"Loaded threshold: {prob_threshold}")
    except FileNotFoundError:
        print("Model artifacts not found.")
        return

    # Generate Candidates
    print("Generating candidates...")
    detector = RecurringDetector(
        field_config=field_config,
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

    feature_cols = [
        'interval_std', 'interval_median', 'amount_cv', 'amount_std',
        'dom_std', 'dow_std', 'count', 'days_span',
        'description_length', 'unique_descriptions'
    ]

    X_test = test_candidates[feature_cols].fillna(0)
    probs = clf.predict_proba(X_test)[:, 1]
    test_candidates['probability'] = probs

    # Map Results
    selected = test_candidates[test_candidates['probability'] >= prob_threshold]

    test_df['recurring_group_id'] = -1
    group_id_counter = 0

    # Map group predictions back to transactions
    group_probs = dict(zip(range(len(selected)), selected['probability']))

    for _, row in selected.iterrows():
        test_df.loc[row['indices'], 'recurring_group_id'] = group_id_counter
        group_id_counter += 1

    # Evaluation
    print("\n--- Test Set Results (Transactional) ---")
    f1 = calculate_f1(test_df)
    print(f"Test F1 Score: {f1:.4f}")

    if field_config.label in test_df.columns:
        y_true = test_df[field_config.label].fillna(0).astype(bool)
        y_pred = test_df['recurring_group_id'] != -1

        # PR-AUC requires continuous scores.
        # Map group probability to transactions. Default 0 for non-grouped.
        # Logic: If txn is in group G, score = prob(G). Else score = 0.

        # 1. Create a series mapping index -> probability
        # Flatten the indices from the candidates
        idx_to_prob = {}
        for _, row in test_candidates.iterrows():
            p = row['probability']
            for idx in row['indices']:
                # If a transaction belongs to multiple candidates (unlikely with DBSCAN but possible),
                # take the max probability
                idx_to_prob[idx] = max(idx_to_prob.get(idx, 0), p)

        y_scores = test_df.index.map(lambda x: idx_to_prob.get(x, 0.0))

        pr_auc = average_precision_score(y_true, y_scores)
        print(f"Test PR AUC: {pr_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    evaluate_test_set()