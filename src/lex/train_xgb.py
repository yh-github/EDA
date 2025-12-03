import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib
from data_loader import load_data, preprocess_data
from recurring_detector import RecurringDetector


def train_xgb():
    print("Loading data...")
    df = load_data()
    df = preprocess_data(df)

    # Split
    account_ids = df['accountId'].unique()
    np.random.seed(42)
    np.random.shuffle(account_ids)

    train_size = int(0.7 * len(account_ids))
    val_size = int(0.15 * len(account_ids))

    train_accounts = account_ids[:train_size]
    val_accounts = account_ids[train_size:train_size + val_size]

    train_df = df[df['accountId'].isin(train_accounts)].copy()
    val_df = df[df['accountId'].isin(val_accounts)].copy()

    print(f"Train accounts: {len(train_accounts)}")
    print(f"Val accounts: {len(val_accounts)}")

    # Generate Candidates (Train)
    print("Generating candidates from Training set...")
    detector = RecurringDetector(
        interval_tolerance=40,
        min_transactions=2,
        amount_cv_threshold=1.0,
        dom_std_threshold=10.0,
        eps=0.29  # Best param from previous tuning
    )

    candidates_df = detector.detect(train_df, return_candidates=True)

    if candidates_df.empty:
        print("No candidates found!")
        return

    print(f"Generated {len(candidates_df)} candidates.")

    # Features
    feature_cols = [
        'interval_std', 'interval_median',
        'amount_cv', 'amount_std',
        'dom_std', 'dow_std',
        'count', 'days_span',
        'description_length', 'unique_descriptions'
    ]

    X_train = candidates_df[feature_cols].fillna(0)
    y_train = candidates_df['label'].astype(int)

    # Train XGBoost
    print("Training XGBoost...")
    import json
    with open('best_params_xgb.json') as f:
        best_params = json.load(f)

    # Extract prob_threshold (not a model param)
    prob_threshold = best_params.pop('prob_threshold', 0.5)
    print(f"Using best params: {best_params}")
    print(f"Prob threshold: {prob_threshold}")

    clf = XGBClassifier(
        **best_params,
        random_state=42,
        scale_pos_weight=1,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Validate
    print("Validating...")
    val_candidates_df = detector.detect(val_df, return_candidates=True)
    if not val_candidates_df.empty:
        X_val = val_candidates_df[feature_cols].fillna(0)
        y_val = val_candidates_df['label'].astype(int)

        # Predict Probabilities
        probs = clf.predict_proba(X_val)[:, 1]
        y_val_pred = (probs >= prob_threshold).astype(int)

        print("Validation Report (Model Level):")
        print(classification_report(y_val, y_val_pred))

    # Save Model
    print("Saving model...")
    joblib.dump(clf, 'recurring_xgb_model.joblib')
    # Save threshold too
    joblib.dump(prob_threshold, 'xgb_threshold.joblib')
    print("Done.")


if __name__ == "__main__":
    train_xgb()
