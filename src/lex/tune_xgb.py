import optuna
import json
from xgboost import XGBClassifier
from lex.data_loader import load_lex_splits
from lex.recurring_detector import RecurringDetector
from lex.analyzer import calculate_f1
from common.config import FieldConfig

# Globals for Objective
global_val_df = None
global_train_candidates = None
global_val_candidates = None
global_feature_cols = None


def objective(trial):
    # Suggest params
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    prob_threshold = trial.suggest_float('prob_threshold', 0.1, 0.9)

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

    # Evaluate on pre-generated validation candidates
    X_val = global_val_candidates[global_feature_cols].fillna(0)
    probs = clf.predict_proba(X_val)[:, 1]

    # Map back to Transactions to calculate transactional F1
    val_candidates_copy = global_val_candidates.copy()
    val_candidates_copy['probability'] = probs
    selected = val_candidates_copy[val_candidates_copy['probability'] >= prob_threshold]

    # Use a clean copy of the dataframe
    val_map_df = global_val_df.copy()
    val_map_df['recurring_group_id'] = -1

    group_id_counter = 0
    for _, row in selected.iterrows():
        indices = row['indices']
        val_map_df.loc[indices, 'recurring_group_id'] = group_id_counter
        group_id_counter += 1

    f1 = calculate_f1(val_map_df)
    return f1


if __name__ == "__main__":
    print("Loading unified splits...")
    field_config = FieldConfig()
    train_df, val_df, _ = load_lex_splits(MultiExpConfig())

    # Preprocess
    train_df = preprocess_lex_features(train_df, field_config)
    global_val_df = preprocess_lex_features(val_df, field_config)

    print("Generating candidates...")
    detector = RecurringDetector(
        field_config=field_config,
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

    print("Starting Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    print(f"Best F1: {study.best_value}")
    print(f"Best params: {study.best_params}")

    with open('best_params_xgb.json', 'w') as f:
        json.dump(study.best_params, f)