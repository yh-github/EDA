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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("pairwise_grouping")


@dataclass
class GroupingConfig:
    window_size: int = 200
    string_threshold: float = 60.0
    match_threshold: float = 0.50
    n_jobs: int = -1


class PairwiseGroupingModel:
    def __init__(self, config: GroupingConfig = GroupingConfig()):
        self.config = config

        # Monotonic Constraints to enforce physical logic:
        # [str_score(+), diff_amt(-), rel_diff_amt(-), is_exact(+), diff_days(0), day_alignment(0)]
        # We leave diff_days and day_alignment unconstrained so the model can learn
        # specific sweet spots (e.g. 7 days, 14 days, 30 days).
        monotonic_cst = [1, -1, -1, 1, 0, 0]

        self.model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_depth=8,
            scoring='average_precision',
            monotonic_cst=monotonic_cst,
            random_state=42
        )

    def _preprocess(self, df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['bankRawDescription'] = df['bankRawDescription'].fillna("").astype(str)
        df['dom'] = df['date'].dt.day

        # CRITICAL: Sort by Account -> Amount -> Date
        # This linearizes the search space for recurring amounts.
        df = df.sort_values(['accountId', 'amount', 'date']).reset_index(drop=True)

        if training:
            # Assign unique IDs to noise so they NEVER match each other
            noise_mask = df['patternId'].isna() | (df['patternId'] == '') | (df['patternId'].astype(str) == '-1')
            df.loc[noise_mask, 'patternId'] = "NOISE_" + df.loc[noise_mask, 'trId'].astype(str)

        return df

    @staticmethod
    def _process_chunk(chunk_df, window_size, string_threshold, training):
        # Extract columns to numpy for raw speed
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

                # Account boundary check
                if acc_ids[i] != acc_ids[j]: break

                # 1. Amount Filter (Fastest Check)
                a1, a2 = amounts[i], amounts[j]
                diff_amt = abs(a1 - a2)

                # Heuristic: If amounts diverge by >$10, stop scanning this window.
                # (Since data is sorted by amount, all subsequent j will be even worse)
                if diff_amt > 10.0 and abs(a1) < 100: break
                if diff_amt > 100.0: break

                # 2. Text Filter
                s1, s2 = texts[i], texts[j]
                str_score = fuzz.token_sort_ratio(s1, s2)
                if str_score < string_threshold: continue

                # 3. Feature Calculation
                max_amt = max(abs(a1), abs(a2)) + 1e-9
                rel_diff_amt = diff_amt / max_amt
                is_exact_amt = 1 if diff_amt < 0.01 else 0

                d1, d2 = dates[i], dates[j]
                diff_days = abs((d1 - d2).astype('timedelta64[D]').astype(int))

                # Circular Day Alignment (0.0 to 1.0)
                # Helps the model distinguish "Regular Monthly" from "Random Habit"
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
                    # Ground Truth: 1 if same patternId, 0 otherwise
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

        logger.info(f"Processing in parallel on {self.config.n_jobs} cores...")
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

        logger.info(f"Generated {len(all_features)} candidate pairs.")

        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int32) if training else None

        # Map Transaction IDs back to DataFrame indices for graph construction
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

        logger.info(f"Training on {len(X)} pairs...")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)

        y_prob = self.model.predict_proba(X_val)[:, 1]
        ap = average_precision_score(y_val, y_prob)
        logger.info(f"Model Validation PR-AUC: {ap:.4f}")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = self._preprocess(df, training=False)
        X, _, pair_indices, df_sorted = self._generate_candidates_parallel(df_clean, training=False)

        if len(X) == 0: return df_sorted

        # 1. Score Pairs
        probs = self.model.predict_proba(X)[:, 1]

        # 2. Filter Weak Links
        mask = probs > self.config.match_threshold
        valid_pairs = np.array(pair_indices)[mask]

        n_nodes = len(df_sorted)
        if len(valid_pairs) == 0:
            df_sorted['pred_group_id'] = -1
            return df_sorted

        # 3. Build Graph & Cluster
        row = valid_pairs[:, 0]
        col = valid_pairs[:, 1]
        data = np.ones(len(valid_pairs))
        adj_matrix = csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))

        n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
        df_sorted['pred_group_id'] = labels

        # 4. Stability Check (The "Anti-Wawa" Filter)
        valid_groups = []
        for gid, grp in df_sorted.groupby('pred_group_id'):
            if len(grp) < 2: continue  # Size 1 is noise

            dates = grp['date'].sort_values()
            diffs = dates.diff().dt.days.dropna()

            if len(diffs) == 0: continue

            median_gap = diffs.median()

            # A. Burst Filter: If everything happened in <4 days average, it's a burst/duplicate
            if median_gap < 4: continue

            # B. Variance Filter:
            # If the gap fluctuates wildly (High Std Dev), it's likely a habit, not a subscription.
            if len(diffs) >= 2:
                gap_std = diffs.std()
                # Tighter tolerance for weekly (short) cycles, looser for monthly
                tolerance = 3.0 if median_gap < 20 else 5.0

                if gap_std > tolerance:
                    continue

            valid_groups.append(gid)

        # Final Cleanup
        df_sorted.loc[~df_sorted['pred_group_id'].isin(valid_groups), 'pred_group_id'] = -1

        return df_sorted


# --- Helper for your environment ---
def load_data(random_state: int | None = None, downsample: float | None = None):
    conf = MultiExpConfig()
    if random_state is not None: conf.random_state = random_state
    if downsample is not None: conf.downsample = downsample
    return load_data_for_config(conf)


if __name__ == "__main__":
    print("Loading data...")
    train_df, val_df, test_df = load_data(random_state=0x5EED, downsample=0.1)

    model = PairwiseGroupingModel()
    model.fit(val_df)

    print("\nRunning Inference...")
    results = model.predict(test_df)

    # Simple Reporting
    recurring = results[results['pred_group_id'] != -1]
    print(f"\nFound {len(recurring)} recurring transactions.")

    if len(recurring) > 0:
        # Show a few examples
        ex_gid = recurring['pred_group_id'].value_counts().index[0]
        print(f"\nExample Group {ex_gid}:")
        print(recurring[recurring['pred_group_id'] == ex_gid][['date', 'amount', 'bankRawDescription']].sort_values(
            'date'))