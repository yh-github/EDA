import joblib
import json
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

from lex.data_loader import load_lex_splits
from lex.recurring_detector import RecurringDetector
from common.config import FieldConfig


def train_xgb():
    print("Initializing configuration...")
    field_config = FieldConfig()

    # Load standardized splits
    train_df, val_df, _ = load_lex_splits()


    print(f"Train transactions: {len(train_df)}")
    print(f"Val transactions:   {len(val_df)}")

    # Generate Candidates
    print("Generating candidates from Training set...")
    detector = RecurringDetector(
        field_config=field_config,
        interval_tolerance=40,
        min_transactions=2,
        amount_cv_threshold=1.0,
        dom_std_threshold=10.0,
        eps=0.29
    )

    candidates_df = detector.detect(train_df, return_candidates=True)

    if candidates_df.empty:
        print("No candidates found! Check data or detector params.")
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

    # Load Params
    try:
        with open('best_params_xgb.json') as f:
            best_params = json.load(f)
    except FileNotFoundError:
        print("best_params_xgb.json not found, using defaults.")
        best_params = {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 100}

    prob_threshold = best_params.pop('prob_threshold', 0.5)

    print("Training XGBoost...")
    clf = XGBClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Validate
    print("Validating...")
    val_candidates_df = detector.detect(val_df, return_candidates=True)

    if not val_candidates_df.empty:
        X_val = val_candidates_df[feature_cols].fillna(0)
        y_val = val_candidates_df['label'].astype(int)

        probs = clf.predict_proba(X_val)[:, 1]
        y_val_pred = (probs >= prob_threshold).astype(int)

        print("Validation Report (Group Level):")
        print(classification_report(y_val, y_val_pred))

    # Save
    print("Saving artifacts...")
    joblib.dump(clf, 'recurring_xgb_model.joblib')
    joblib.dump(prob_threshold, 'xgb_threshold.joblib')
    print("Done.")


if __name__ == "__main__":
    train_xgb()