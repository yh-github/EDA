import json

import optuna
import pandas as pd
from sklearn.metrics import precision_score, recall_score, average_precision_score
from xgboost import XGBClassifier

from common.config import FieldConfig
from lex.data_loader import load_lex_splits, preprocess_lex_features
from lex.recurring_detector import RecurringDetector

# Globals for Objective
global_val_df = None
global_train_candidates = None
global_val_candidates = None
global_feature_cols = None
global_field_config = None


def objective(trial):
    # Refined Search Space based on previous logs (Best F1 ~0.80 at depth 8, lr 0.05)
    max_depth = trial.suggest_int('max_depth', 5, 12)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.15, log=True)
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
    prob_threshold = trial.suggest_float('prob_threshold', 0.3, 0.7)

    clf = XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        n_jobs=-1
    )

    # Train
    X_train = global_train_candidates[global_feature_cols].fillna(0)
    y_train = global_train_candidates['label'].astype(int)
    clf.fit(X_train, y_train)

    # --- Evaluation ---
    # 1. Predict Probs on Candidates
    X_val = global_val_candidates[global_feature_cols].fillna(0)
    probs = clf.predict_proba(X_val)[:, 1]

    # 2. Binary Mapping (Thresholded)
    val_candidates_copy = global_val_candidates.copy()
    val_candidates_copy['probability'] = probs

    # Select groups passing threshold
    selected = val_candidates_copy[val_candidates_copy['probability'] >= prob_threshold]

    # Map back to Transactions
    # We default everything to 0 (Not Recurring)
    y_pred = pd.Series(0, index=global_val_df.index)

    # Mark recurring transactions
    for _, row in selected.iterrows():
        y_pred.loc[row['indices']] = 1

    # 3. Ground Truth
    y_true = global_val_df[global_field_config.label].fillna(0).astype(int)

    # 4. Calculate Binary Metrics (P, R, F1)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    # 5. Calculate PR-AUC (Needs continuous scores)
    # We map the group probability to every transaction in that group.
    # Transactions not in any candidate group get 0.0.

    # Flatten the mapping: idx -> prob
    idx_prob_map = {}
    for idx, row in val_candidates_copy.iterrows():
        p = row['probability']
        for txn_idx in row['indices']:
            if txn_idx in idx_prob_map:
                idx_prob_map[txn_idx] = max(idx_prob_map[txn_idx], p)
            else:
                idx_prob_map[txn_idx] = p

    # Update y_scores safely
    y_scores = global_val_df.index.map(lambda x: idx_prob_map.get(x, 0.0))

    pr_auc = average_precision_score(y_true, y_scores)

    # Store metrics in trial for analysis
    trial.set_user_attr("precision", precision)
    trial.set_user_attr("recall", recall)
    trial.set_user_attr("pr_auc", pr_auc)

    # Print for user visibility
    print(f"Trial {trial.number}: F1={f1:.4f} | P={precision:.4f} | R={recall:.4f} | AUC={pr_auc:.4f}")

    return f1


if __name__ == "__main__":
    print("Loading unified splits...")
    global_field_config = FieldConfig()
    train_df, global_val_df, _ = load_lex_splits()

    print("Generating candidates...")
    detector = RecurringDetector(
        field_config=global_field_config,
        interval_tolerance=40,
        min_transactions=2,
        amount_cv_threshold=1.0,
        dom_std_threshold=10.0,
        eps=0.29
    )

    global_train_candidates = detector.detect(train_df, return_candidates=True)
    global_val_candidates = detector.detect(global_val_df, return_candidates=True)

    if global_train_candidates.empty or global_val_candidates.empty:
        print("Not enough candidates for tuning.")
        exit()

    global_feature_cols = [
        'interval_std', 'interval_median', 'amount_cv', 'amount_std',
        'dom_std', 'dow_std', 'count', 'days_span',
        'description_length', 'unique_descriptions'
    ]

    print("Starting Optuna (Refined Search)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)  # Increased trials

    print("=" * 60)
    print(f"Best F1: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")
    print("=" * 60)

    # Print full metrics for best trial
    best_trial = study.best_trial
    print(f"Best P: {best_trial.user_attrs.get('precision', 0):.4f}")
    print(f"Best R: {best_trial.user_attrs.get('recall', 0):.4f}")
    print(f"Best PR-AUC: {best_trial.user_attrs.get('pr_auc', 0):.4f}")

    with open('best_params_xgb.json', 'w') as f:
        json.dump(study.best_params, f)