import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from rapidfuzz import fuzz
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from joblib import Parallel, delayed, cpu_count

from multi.config import MultiExpConfig
from multi.reload_utils import load_data_for_config

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("pairwise")


@dataclass
class PairwiseConfig:
    window_size: int = 200
    string_threshold: float = 60.0
    match_threshold: float = 0.50  # Slightly relaxed to catch "noisy" dates
    n_jobs: int = -1


class PairwiseRecurrenceModel:
    def __init__(self, config: PairwiseConfig = PairwiseConfig()):
        self.config = config

        # REMOVED monotonic constraint on day_alignment (Index 5).
        # This allows the model to learn that "Bad Alignment" is fine for
        # bi-monthly (1st/15th) or weekly (Mon/Mon) patterns differently.
        # Features: [str_score, diff_amt, rel_diff_amt, is_exact, diff_days, day_alignment]
        monotonic_cst = [1, -1, -1, 1, 0, 0]

        self.model = HistGradientBoostingClassifier(
            max_iter=200,  # More trees to learn complex cycle rules
            learning_rate=0.05,  # Slower learning for better generalization
            max_depth=8,  # Deeper trees to capture interaction (DayDiff vs Alignment)
            scoring='average_precision',
            monotonic_cst=monotonic_cst,
            random_state=42
        )

    def _preprocess(self, df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['bankRawDescription'] = df['bankRawDescription'].fillna("").astype(str)
        df['dom'] = df['date'].dt.day

        # Global Sort
        df = df.sort_values(['accountId', 'amount', 'date']).reset_index(drop=True)

        if training:
            # Handle NaN patternIds as unique noise
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
        texts = chunk_df['bankRawDescription'].values

        pids = None
        if training:
            pids = chunk_df['patternId'].values

        n = len(chunk_df)
        features = []
        labels = []
        edges = []

        for i in range(n):
            for offset in range(1, window_size + 1):
                j = i + offset
                if j >= n: break
                if acc_ids[i] != acc_ids[j]: break

                # Amount Filter (Heuristic)
                a1, a2 = amounts[i], amounts[j]
                diff_amt = abs(a1 - a2)
                if diff_amt > 10.0 and abs(a1) < 100: break
                if diff_amt > 100.0: break

                # Text Filter
                s1, s2 = texts[i], texts[j]
                str_score = fuzz.token_sort_ratio(s1, s2)
                if str_score < string_threshold: continue

                # Features
                max_amt = max(abs(a1), abs(a2)) + 1e-9
                rel_diff_amt = diff_amt / max_amt
                is_exact_amt = 1 if diff_amt < 0.01 else 0

                d1, d2 = dates[i], dates[j]
                diff_days = abs((d1 - d2).astype('timedelta64[D]').astype(int))

                # Circular Day Alignment (0..1)
                # 1.0 = Same day (e.g. 5th and 5th)
                # 0.0 = Opposite side of month (e.g. 1st and 15th)
                day_diff = abs(doms[i] - doms[j])
                circular_dist = min(day_diff, 30 - day_diff)
                day_alignment = 1.0 - (circular_dist / 15.0)

                features.append([
                    str_score,
                    diff_amt,
                    rel_diff_amt,
                    is_exact_amt,
                    diff_days,
                    day_alignment
                ])

                edges.append((ids[i], ids[j]))

                if training:
                    # Match if PatternIDs are identical (and not Noise)
                    is_match = 1 if (pids[i] == pids[j]) else 0
                    labels.append(is_match)

        return features, labels, edges

    def _generate_candidates_parallel(self, df: pd.DataFrame, training: bool = False):
        unique_accounts = df['accountId'].unique()
        n_chunks = max(1, cpu_count() * 2)
        account_chunks = np.array_split(unique_accounts, n_chunks)

        tasks = []
        for acc_chunk in account_chunks:
            if len(acc_chunk) == 0: continue
            subset = df[df['accountId'].isin(acc_chunk)].copy()
            tasks.append(subset)

        results = Parallel(n_jobs=self.config.n_jobs)(
            delayed(self._process_chunk)(
                chunk,
                self.config.window_size,
                self.config.string_threshold,
                training
            ) for chunk in tasks
        )

        all_features = []
        all_labels = []
        all_edges = []

        for feat, lab, edge in results:
            if feat:
                all_features.extend(feat)
                all_edges.extend(edge)
                if training:
                    all_labels.extend(lab)

        logger.info(f"Generated {len(all_features)} pairs.")

        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int32) if training else None

        if len(all_edges) > 0:
            id_to_idx = {tr_id: idx for idx, tr_id in enumerate(df['trId'])}
            pair_indices = []
            for u, v in all_edges:
                if u in id_to_idx and v in id_to_idx:
                    pair_indices.append((id_to_idx[u], id_to_idx[v]))
        else:
            pair_indices = []

        return X, y, pair_indices, df

    def fit(self, df: pd.DataFrame):
        df_clean = self._preprocess(df, training=True)
        X, y, _, _ = self._generate_candidates_parallel(df_clean, training=True)

        if len(X) == 0: return

        print(f"Training on {len(X)} pairs...")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)

        y_prob = self.model.predict_proba(X_val)[:, 1]
        ap = average_precision_score(y_val, y_prob)
        print(f"--- Validation Results ---")
        print(f"Average Precision (PR-AUC): {ap:.4f}")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = self._preprocess(df, training=False)
        X, _, pair_indices, df_sorted = self._generate_candidates_parallel(df_clean, training=False)

        if len(X) == 0: return df_sorted

        probs = self.model.predict_proba(X)[:, 1]
        mask = probs > self.config.match_threshold
        valid_pairs = np.array(pair_indices)[mask]

        n_nodes = len(df_sorted)
        # Handle empty case
        if len(valid_pairs) == 0:
            df_sorted['pred_group_id'] = -1
            df_sorted['cycle_type'] = 'None'
            return df_sorted

        # Graph Clustering
        row = valid_pairs[:, 0]
        col = valid_pairs[:, 1]
        data = np.ones(len(valid_pairs))
        adj_matrix = csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))

        n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
        df_sorted['pred_group_id'] = labels

        # --- Post-Process: Cycle Classification ---
        cycle_map = {}

        for gid, grp in df_sorted.groupby('pred_group_id'):
            if len(grp) < 2:
                cycle_map[gid] = 'None'
                continue

            dates = grp['date'].sort_values()
            diffs = dates.diff().dt.days.dropna()
            median_gap = diffs.median()

            # Simple Burst Filter
            if median_gap < 4:
                cycle_map[gid] = 'None'
                continue

            cycle_map[gid] = self.classify_cycle(median_gap)

        df_sorted['cycle_type'] = df_sorted['pred_group_id'].map(cycle_map).fillna('None')

        # Set non-recurring to -1
        df_sorted.loc[df_sorted['cycle_type'] == 'None', 'pred_group_id'] = -1

        return df_sorted

    @staticmethod
    def classify_cycle(gap: float) -> str:
        """Maps a median day gap to your business labels."""
        if 5.5 <= gap <= 8.5:
            return 'onceAWeek'
        elif 12.0 <= gap <= 16.5:
            # Covers 14 days (bi-weekly) and 15 days (twice a month)
            # Hard to distinguish purely on median without variance check,
            # but usually grouped together or distinguished by day-of-week alignment.
            # For now, we can call it 'twiceAMonth' or 'onceEvery2Weeks' based on preference.
            if 14.5 <= gap <= 16.0:
                return 'twiceAMonth'
            return 'onceEvery2Weeks'
        elif 26.0 <= gap <= 33.0:
            # 28 days (4 weeks) vs 30/31 days (Monthly)
            if gap < 29.0:
                return 'onceEvery4Weeks'
            return 'monthly'
        elif 33.0 < gap <= 45.0:
            return 'every4_5Weeks'
        else:
            return 'weekBasedOther'


if __name__ == "__main__":
    print("Loading data...")
    train_df, val_df, test_df = load_data_for_config(MultiExpConfig())

    model = PairwiseRecurrenceModel()
    model.fit(val_df)

    print("\nRunning Inference...")
    results = model.predict(test_df)

    # Show Results with Cycle Labels
    recurring = results[results['pred_group_id'] != -1]
    print(f"\nFound {len(recurring)} recurring transactions.")

    # Check for specific complex cycles
    for c_type in ['twiceAMonth', 'onceEvery4Weeks', 'every4_5Weeks', 'monthly']:
        subset = recurring[recurring['cycle_type'] == c_type]
        if not subset.empty:
            gid = subset.iloc[0]['pred_group_id']
            print(f"\n--- Example {c_type} (Group {gid}) ---")
            print(
                subset[subset['pred_group_id'] == gid][['date', 'amount', 'bankRawDescription']].to_string(index=False))