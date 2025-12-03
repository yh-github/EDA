# Recurring Transaction Detector System Documentation

## 1. System Overview
The system uses a **Hybrid Two-Stage Approach** to detect recurring transactions (subscriptions, bills, salary, etc.) in bank transaction data.

1.  **Stage 1: Candidate Generation (Heuristic/Unsupervised)**
    *   Transactions are grouped by `accountId`.
    *   **Clustering**: Transactions with similar descriptions are clustered using **TF-IDF + DBSCAN** (or fuzzy matching).
    *   **Filtering**: Only clusters with $\ge 2$ transactions are kept as "Candidate Groups".
    *   **Feature Extraction**: Statistical features (interval stability, amount stability) are calculated for each candidate group.

2.  **Stage 2: Classification (Supervised ML)**
    *   An **XGBoost Classifier** evaluates each candidate group.
    *   It predicts the probability that the group represents a true recurring pattern.
    *   Groups with probability $>$ Threshold (tuned) are flagged as `recurring`.

## 2. Feature Engineering
For each candidate group, we extract the following features:

| Feature | Description |
| :--- | :--- |
| `interval_std` | Standard deviation of days between transactions. (Lower = More Periodic) |
| `interval_median` | Median days between transactions (e.g., 30 for monthly). |
| `amount_cv` | Coefficient of Variation of amounts ($\sigma / \mu$). (Lower = More Stable) |
| `amount_std` | Standard deviation of amounts. |
| `dom_std` | Standard deviation of the Day of Month (e.g., always on the 15th). |
| `dow_std` | Standard deviation of the Day of Week (useful for weekly patterns). |
| `count` | Number of transactions in the group. |
| `days_span` | Total days between first and last transaction. |
| `description_length` | Average length of the transaction description. |
| `unique_descriptions` | Number of unique raw descriptions in the cluster. |

## 3. Model Training & Validation

### Data Split
To ensure robust performance and prevent leakage, data is split by **Account ID**:
*   **Train**: 70% of accounts
*   **Validation**: 15% of accounts (Used for Hyperparameter Tuning)
*   **Test**: 15% of accounts (Held-out for final evaluation)

### Model: XGBoost
*   **Algorithm**: XGBoost Classifier
*   **Objective**: Binary Classification (Is this group recurring?)
*   **Hyperparameters**: Tuned using Optuna on the Validation set.
*   **Thresholding**: The probability threshold is optimized to maximize the F1 Score on the Validation set.

## 4. Performance Metrics
Performance on the **Held-out Test Set** (Unseen Data):

| Metric | Score |
| :--- | :--- |
| **F1 Score** | **0.80** |
| **PR AUC** | **0.78** |
| Precision | 0.82 |
| Recall | 0.78 |
| Accuracy | 0.94 |

*Note: Metrics are calculated per-transaction.*

## 5. Output Format
The system generates the following outputs for each transaction:
1.  **`recurring_group_id`**: A unique integer ID for the recurring pattern (e.g., `101`). If not recurring, this is `-1`.
2.  **`recurring_probability`**: The model's confidence score (0.0 to 1.0) that this transaction belongs to a recurring group.
3.  **`is_recurring`**: A binary label (True/False) derived from the probability threshold.

## 6. Repository Structure
The following files should be included in the repository:

**Core Logic:**
*   [recurring_detector.py](file:///c:/Work/new_code/recurring_detector.py): Main class for candidate generation and heuristic logic.
*   [data_loader.py](file:///c:/Work/new_code/data_loader.py): Utilities for loading and preprocessing data.
*   [analyzer.py](file:///c:/Work/new_code/analyzer.py): Helper functions for calculating metrics (F1, etc.).

**Training & Evaluation:**
*   [train_xgb.py](file:///c:/Work/new_code/train_xgb.py): Script to train the XGBoost model and save artifacts.
*   [tune_xgb.py](file:///c:/Work/new_code/tune_xgb.py): Script to tune hyperparameters using Optuna.
*   [evaluate_test_set.py](file:///c:/Work/new_code/evaluate_test_set.py): Script to evaluate performance on the held-out test set.

**Models & Config:**
*   [recurring_xgb_model.joblib](file:///c:/Work/new_code/recurring_xgb_model.joblib): The trained XGBoost model.
*   [xgb_threshold.joblib](file:///c:/Work/new_code/xgb_threshold.joblib): The optimized probability threshold.
*   [best_params_xgb.json](file:///c:/Work/new_code/best_params_xgb.json): Best hyperparameters found during tuning.

**Dependencies:**
*   `requirements.txt`: (Should be created) `pandas`, `numpy`, `xgboost`, `scikit-learn`, `optuna`, [joblib](file:///c:/Work/new_code/xgb_threshold.joblib).

## 7. How to Run

### Training
To retrain the model on new data:
```bash
python train_xgb.py
```
This will:
1. Load data from [all_data.csv](file:///c:/Work/new_code/all_data.csv).
2. Split into Train/Val.
3. Train XGBoost.
4. Save model to [recurring_xgb_model.joblib](file:///c:/Work/new_code/recurring_xgb_model.joblib).

### Inference (Detection)
To detect recurring transactions in a new dataframe `df`:

```python
import joblib
from lex.recurring_detector import RecurringDetector

# 1. Load Model
clf = joblib.load('recurring_xgb_model.joblib')
threshold = joblib.load('xgb_threshold.joblib')

# 2. Generate Candidates
detector = RecurringDetector()
candidates = detector.detect(df, return_candidates=True)

# 3. Predict
features = candidates[['interval_std', 'amount_cv', ...]] # Ensure cols match
probs = clf.predict_proba(features)[:, 1]

# 4. Filter & Map
selected = candidates[probs >= threshold]
# Map back to transactions...
```
