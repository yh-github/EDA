import optuna
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from data_loader import load_data, preprocess_data
from recurring_detector import RecurringDetector
from analyzer import calculate_f1


def objective(trial):
    global val_df, train_candidates, feature_cols

    # Suggest params
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    prob_threshold = trial.suggest_float('prob_threshold', 0.1, 0.9)

    # Train Model
    clf = XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        n_jobs=-1
    )

    X_train = train_candidates[feature_cols].fillna(0)
    y_train = train_candidates['label'].astype(int)

    clf.fit(X_train, y_train)

    # Evaluate on Val
    # We need to run detection on val_df using the classifier
    # To save time, we can pre-generate val candidates once globally if we assume fixed detector params
    # But detector params (eps) were tuned for RF. Let's assume they are good for XGB too for now.

    global val_candidates
    X_val = val_candidates[feature_cols].fillna(0)
    probs = clf.predict_proba(X_val)[:, 1]

    val_candidates_copy = val_candidates.copy()
    val_candidates_copy['probability'] = probs

    selected = val_candidates_copy[val_candidates_copy['probability'] >= prob_threshold]

    # Map back
    val_df_copy = val_df.copy()
    val_df_copy['recurring_group_id'] = -1

    group_id_counter = 0
    for _, row in selected.iterrows():
        indices = row['indices']
        val_df_copy.loc[indices, 'recurring_group_id'] = group_id_counter
        group_id_counter += 1

    f1 = calculate_f1(val_df_copy)
    return f1


if __name__ == "__main__":
    print("Loading data for tuning...")
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

    # Pre-generate candidates to speed up tuning loop
    print("Generating candidates...")
    detector = RecurringDetector(
        interval_tolerance=40,
        min_transactions=2,
        amount_cv_threshold=1.0,
        dom_std_threshold=10.0,
        eps=0.29
    )

    train_candidates = detector.detect(train_df, return_candidates=True)
    val_candidates = detector.detect(val_df, return_candidates=True)

    feature_cols = [
        'interval_std', 'interval_median',
        'amount_cv', 'amount_std',
        'dom_std', 'dow_std',
        'count', 'days_span',
        'description_length', 'unique_descriptions'
    ]

    print("Starting Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    print("Best params:")
    print(study.best_params)
    print(f"Best F1: {study.best_value}")

    import json

    with open('best_params_xgb.json', 'w') as f:
        json.dump(study.best_params, f)
