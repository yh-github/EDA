import logging
import re
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from rapidfuzz import fuzz
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from multi.config import MultiExpConfig
from multi.reload_utils import load_data_for_config

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("pairwise_tuner")


# --- 1. Nuanced Text Processing ---
def check_recurrence_keywords(text: str) -> int:
    """Checks for explicit recurrence signals before we clean them away."""
    if not isinstance(text, str): return 0
    text = text.lower()
    # "fee" is often recurring (maintenance fee), "recurring" is obvious
    if re.search(r'\b(recurring|monthly|annual|fee|subscription|autopay)\b', text):
        return 1
    return 0


def clean_description(text: str) -> str:
    """
    Removes banking noise to isolate the MERCHANT NAME for string matching.
    """
    if not isinstance(text, str): return ""
    text = text.lower()

    # We remove these for MATCHING, but we captured the signal in 'check_recurrence_keywords'
    noise_patterns = [
        r'pos purchase', r'purchase', r'recurring', r'terminal',
        r'authorized transfer', r'check card', r'debit card',
        r'payment', r'ach', r'wd', r'withdrawal', r'deposit',
        r'transaction', r'\d{2}/\d{2}',  # dates
        r'\d{3}-\d{3}-\d{4}',  # phones
        r'\*+',  # masks
        r'\b\d+\b'  # standalone digits (store IDs)
    ]

    for pat in noise_patterns:
        text = re.sub(pat, ' ', text)

    return re.sub(r'\s+', ' ', text).strip()


@dataclass
class HyperParams:
    window_size: int = 200
    string_threshold: float = 60.0
    learning_rate: float = 0.05
    max_depth: int = 6
    max_iter: int = 200
    l2_regularization: float = 0.0
    match_threshold: float = 0.50


class PairwiseRecurrenceSystem:
    def __init__(self, params: HyperParams):
        self.params = params

        # Features: [str_score, diff_amt, rel_diff, is_exact, diff_days, day_align, keyword_match]
        # Added Index 6 (keyword_match) with Positive Constraint (1)
        monotonic_cst = [1, -1, -1, 1, 0, 0, 1]

        self.model = HistGradientBoostingClassifier(
            max_iter=params.max_iter,
            learning_rate=params.learning_rate,
            max_depth=params.max_depth,
            l2_regularization=params.l2_regularization,
            scoring='average_precision',
            monotonic_cst=monotonic_cst,
            random_state=42
        )
        self.feature_names = [
            "str_score", "diff_amt", "rel_diff",
            "is_exact", "diff_days", "day_align", "has_keyword"
        ]

    def _preprocess(self, df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # 1. Extract Signal
        df['has_keyword'] = df['bankRawDescription'].apply(check_recurrence_keywords)

        # 2. Clean for Matching
        df['clean_text'] = df['bankRawDescription'].fillna("").astype(str).apply(clean_description)
        df['dom'] = df['date'].dt.day

        # Sort
        df = df.sort_values(['accountId', 'amount', 'date']).reset_index(drop=True)

        if training:
            noise_mask = df['patternId'].isna() | (df['patternId'] == '') | (df['patternId'].astype(str) == '-1')
            df.loc[noise_mask, 'patternId'] = "NOISE_" + df.loc[noise_mask, 'trId'].astype(str)

        return df

    @staticmethod
    def _process_chunk(chunk_df, window_size, string_threshold, training):
        ids = chunk_df['trId'].values
        acc_ids = chunk_df['accountId'].values
        amounts = chunk_df['amount'].values.astype(float)
        dates = chunk_df['date'].values
        doms = chunk_df['dom'].values
        texts = chunk_df['clean_text'].values
        keywords = chunk_df['has_keyword'].values  # [0, 1]

        pids = None if not training else chunk_df['patternId'].values
        n = len(chunk_df)

        features = []
        labels = []
        meta_pairs = []

        for i in range(n):
            for offset in range(1, window_size + 1):
                j = i + offset
                if j >= n: break
                if acc_ids[i] != acc_ids[j]: break

                a1, a2 = amounts[i], amounts[j]
                diff_amt = abs(a1 - a2)
                if diff_amt > 10.0 and abs(a1) < 100: break
                if diff_amt > 100.0: break

                s1, s2 = texts[i], texts[j]
                str_score = fuzz.token_sort_ratio(s1, s2)
                if str_score < string_threshold: continue

                max_amt = max(abs(a1), abs(a2)) + 1e-9
                rel_diff = diff_amt / max_amt
                is_exact = 1 if diff_amt < 0.01 else 0

                d1, d2 = dates[i], dates[j]
                diff_days = abs((d1 - d2).astype('timedelta64[D]').astype(int))

                day_diff = abs(doms[i] - doms[j])
                circular = min(day_diff, 30 - day_diff)
                day_align = 1.0 - (circular / 15.0)

                # New Feature: Do BOTH have a keyword? Or at least one?
                # We use sum (0, 1, or 2). Monotonic constraint will handle it.
                keyword_score = keywords[i] + keywords[j]

                features.append([
                    str_score, diff_amt, rel_diff, is_exact,
                    diff_days, day_align, keyword_score
                ])

                if training:
                    is_match = 1 if (pids[i] == pids[j]) else 0
                    labels.append(is_match)
                    meta_pairs.append((i, j))

        return features, labels, meta_pairs

    def generate_candidates(self, df: pd.DataFrame, training: bool = False):
        unique_accounts = df['accountId'].unique()
        n_chunks = max(1, cpu_count())
        chunks = np.array_split(unique_accounts, n_chunks)

        tasks = [df[df['accountId'].isin(c)].copy() for c in chunks if len(c) > 0]

        results = Parallel(n_jobs=-1)(
            delayed(self._process_chunk)(
                chunk, self.params.window_size, self.params.string_threshold, training
            ) for chunk in tasks
        )

        X_list, y_list, valid_rows_i, valid_rows_j = [], [], [], []

        for task_df, (feats, labs, pairs) in zip(tasks, results):
            if not feats: continue
            X_list.extend(feats)
            y_list.extend(labs)

            if training:
                # Reconstruct rows for error analysis
                task_df_reset = task_df.reset_index(drop=True)
                pairs_arr = np.array(pairs)
                if len(pairs_arr) > 0:
                    valid_rows_i.append(task_df_reset.iloc[pairs_arr[:, 0]])
                    valid_rows_j.append(task_df_reset.iloc[pairs_arr[:, 1]])

        if not X_list: return None, None, None

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        meta_df = None
        if training and valid_rows_i:
            df_i = pd.concat(valid_rows_i).reset_index(drop=True).add_suffix('_1')
            df_j = pd.concat(valid_rows_j).reset_index(drop=True).add_suffix('_2')
            meta_df = pd.concat([df_i, df_j], axis=1)

        return X, y, meta_df

    def train_and_evaluate(self, df_train: pd.DataFrame, trial=None):
        df_clean = self._preprocess(df_train, training=True)
        X, y, meta_df = self.generate_candidates(df_clean, training=True)

        if X is None: return 0.0

        X_tr, X_val, y_tr, y_val, meta_tr, meta_val = train_test_split(
            X, y, meta_df, test_size=0.2, random_state=42
        )

        self.model.fit(X_tr, y_tr)

        y_prob = self.model.predict_proba(X_val)[:, 1]
        ap = average_precision_score(y_val, y_prob)

        if trial is None:
            self._log_diagnostics(y_val, y_prob, meta_val, X_val)

        return ap

    def _log_diagnostics(self, y_true, y_prob, meta_df, X_val):
        logger.info("\n--- FEATURE IMPORTANCE ---")
        r = permutation_importance(self.model, X_val[:10000], y_true[:10000], n_repeats=5, random_state=42)
        for i in r.importances_mean.argsort()[::-1]:
            logger.info(f"{self.feature_names[i]:<15}: {r.importances_mean[i]:.4f}")

        logger.info("\n--- MISTAKE ANALYSIS ---")
        res = meta_df.copy()
        res['prob'] = y_prob
        res['true'] = y_true

        fp = res[(res['true'] == 0) & (res['prob'] > 0.8)].sort_values('prob', ascending=False)
        if not fp.empty:
            logger.info(f"Top False Positives:")
            cols = ['clean_text_1', 'amount_1', 'date_1', 'clean_text_2', 'prob']
            print(fp[cols].head(5).to_string(index=False))

        fn = res[(res['true'] == 1) & (res['prob'] < 0.2)].sort_values('prob', ascending=True)
        if not fn.empty:
            logger.info(f"\nTop False Negatives:")
            print(fn[cols].head(5).to_string(index=False))


def objective(trial):
    params = HyperParams(
        window_size=trial.suggest_categorical("window_size", [200]),
        string_threshold=trial.suggest_int("string_thresh", 55, 70),
        learning_rate=trial.suggest_float("lr", 0.05, 0.2),
        max_depth=trial.suggest_int("depth", 4, 8)
    )

    train_df, _, _ = load_data_for_config(MultiExpConfig())
    # Subsample for faster tuning if needed
    sys = PairwiseRecurrenceSystem(params)
    return sys.train_and_evaluate(train_df, trial=trial)


if __name__ == "__main__":
    logger.info("Starting Optuna Tuning...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    print("\nBest Params:", study.best_params)

    logger.info("Running Final Analysis...")
    best = study.best_params
    final_params = HyperParams(
        window_size=best['window_size'],
        string_threshold=best['string_thresh'],
        learning_rate=best['lr'],
        max_depth=best['depth']
    )

    train_df, _, _ = load_data_for_config(MultiExpConfig())
    sys = PairwiseRecurrenceSystem(final_params)
    sys.train_and_evaluate(train_df, trial=None)