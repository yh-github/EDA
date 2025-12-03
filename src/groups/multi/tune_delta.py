import logging
import re
import pandas as pd
import numpy as np

from dataclasses import dataclass
from rapidfuzz import fuzz
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from joblib import Parallel, delayed, cpu_count
from sklearn.inspection import permutation_importance

from multi.config import MultiExpConfig
from multi.reload_utils import load_data_for_config

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("label_cleaner")


def check_recurrence_keywords(text: str) -> int:
    if not isinstance(text, str): return 0
    text = text.lower()
    if re.search(r'\b(recurring|monthly|annual|fee|subscription|autopay)\b', text):
        return 1
    return 0


def clean_description(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    noise_patterns = [
        r'pos purchase', r'purchase', r'recurring', r'terminal',
        r'authorized transfer', r'check card', r'debit card',
        r'payment', r'ach', r'wd', r'withdrawal', r'deposit',
        r'transaction', r'\d{2}/\d{2}', r'\d{3}-\d{3}-\d{4}',
        r'\*+', r'\b\d+\b'
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


class PairwiseRecurrenceSystem:
    def __init__(self, params: HyperParams):
        self.params = params

        # Features:
        # [str_score(+), diff_amt(-), rel_diff(-), is_exact(+),
        #  diff_days(0), day_align(0), keyword(+), sign_match(+)]
        monotonic_cst = [1, -1, -1, 1, 0, 0, 1, 1]

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
            "str_score", "diff_amt", "rel_diff", "is_exact",
            "diff_days", "day_align", "has_keyword", "sign_match"
        ]

    def _preprocess(self, df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['has_keyword'] = df['bankRawDescription'].apply(check_recurrence_keywords)
        df['clean_text'] = df['bankRawDescription'].fillna("").astype(str).apply(clean_description)
        df['dom'] = df['date'].dt.day

        # Sort for windowing
        df = df.sort_values(['accountId', 'amount', 'date']).reset_index(drop=True)

        if training:
            noise_mask = df['patternId'].isna() | (df['patternId'] == '') | (df['patternId'].astype(str) == '-1')
            df.loc[noise_mask, 'patternId'] = "NOISE_" + df.loc[noise_mask, 'trId'].astype(str)

        return df

    @staticmethod
    def _process_chunk(chunk_df, window_size, string_threshold, training):
        # Extract columns
        acc_ids = chunk_df['accountId'].values
        amounts = chunk_df['amount'].values.astype(float)
        dates = chunk_df['date'].values
        doms = chunk_df['dom'].values
        texts = chunk_df['clean_text'].values
        keywords = chunk_df['has_keyword'].values

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

                # NEW: Sign Check (1 if signs match, 0 if different)
                # This prevents matching a Refund (-$27) with a Fee (+$27)
                sign_match = 1 if np.sign(a1) == np.sign(a2) else 0

                # Heuristic Filters
                diff_amt = abs(a1 - a2)
                if diff_amt > 10.0 and abs(a1) < 100: break
                if diff_amt > 100.0: break

                s1, s2 = texts[i], texts[j]
                str_score = fuzz.token_sort_ratio(s1, s2)
                if str_score < string_threshold: continue

                # Feature Calc
                max_amt = max(abs(a1), abs(a2)) + 1e-9
                rel_diff = diff_amt / max_amt
                is_exact = 1 if diff_amt < 0.01 else 0

                d1, d2 = dates[i], dates[j]
                diff_days = abs((d1 - d2).astype('timedelta64[D]').astype(int))

                day_diff = abs(doms[i] - doms[j])
                circular = min(day_diff, 30 - day_diff)
                day_align = 1.0 - (circular / 15.0)

                keyword_score = keywords[i] + keywords[j]

                features.append([
                    str_score, diff_amt, rel_diff, is_exact,
                    diff_days, day_align, keyword_score, sign_match
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
            self._log_mistakes(y_val, y_prob, meta_val, X_val)

        return ap

    def _log_mistakes(self, y_true, y_prob, meta_df, X_val):
        logger.info("\n--- FEATURE IMPORTANCE ---")
        r = permutation_importance(self.model, X_val[:5000], y_true[:5000], n_repeats=3, random_state=42)
        for i in r.importances_mean.argsort()[::-1]:
            logger.info(f"{self.feature_names[i]:<15}: {r.importances_mean[i]:.4f}")

        res = meta_df.copy()
        res['prob'] = y_prob
        res['true'] = y_true

        # Identify "Potential Missing Labels" (High prob, Truth=0)
        # We output AccountID and TrID for human review
        fp = res[(res['true'] == 0) & (res['prob'] > 0.90)].sort_values('prob', ascending=False)

        logger.info(f"\n--- POTENTIAL MISSING LABELS ({len(fp)} pairs found) ---")
        logger.info("These pairs have >90% probability but are labeled as NO match.")

        # Columns for human labeler
        cols = ['accountId_1', 'trId_1', 'date_1', 'amount_1', 'bankRawDescription_1',
                'trId_2', 'date_2', 'amount_2', 'bankRawDescription_2', 'prob']

        if not fp.empty:
            print(fp[cols].head(10).to_string(index=False))
            # Save to CSV for the user
            fp[cols].to_csv("potential_missing_labels.csv", index=False)
            logger.info("Saved full list to 'potential_missing_labels.csv'")


# Use best known params for the cleaning run
def run_cleaning_pass():
    params = HyperParams(
        window_size=200,
        string_threshold=60,
        learning_rate=0.08,
        max_depth=6
    )

    train_df, _, _ = load_data_for_config(MultiExpConfig())
    # Run on FULL data to find all errors
    sys = PairwiseRecurrenceSystem(params)
    sys.train_and_evaluate(train_df, trial=None)


if __name__ == "__main__":
    run_cleaning_pass()