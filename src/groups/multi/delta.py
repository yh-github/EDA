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

from multi.config import MultiExpConfig
from multi.reload_utils import load_data_for_config

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("pairwise")


@dataclass
class PairwiseConfig:
    window_size: int = 20  # Look at 20 neighbors in the sorted-amount list
    string_threshold: float = 60.0  # Ignore pairs with <60% string match (speedup)
    match_threshold: float = 0.55  # Probability threshold to declare a "Link"


class PairwiseRecurrenceModel:
    def __init__(self, config: PairwiseConfig = PairwiseConfig()):
        self.config = config
        # HistGradientBoosting is fast, handles NaNs, and is robust for tabular data
        self.model = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            max_depth=6,
            scoring='average_precision',
            random_state=42
        )

    def _preprocess(self, df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
        df = df.copy()

        # 1. Standardize Date
        df['date'] = pd.to_datetime(df['date'])

        # 2. Ensure Text is String
        df['bankRawDescription'] = df['bankRawDescription'].fillna("").astype(str)

        # 3. Handle Labels for Training
        if training:
            # If patternId is NaN, it's noise.
            # We give it a UNIQUE ID so it never matches anything else (including other noise).
            noise_mask = df['patternId'].isna() | (df['patternId'] == '')
            df.loc[noise_mask, 'patternId'] = "NOISE_" + df.loc[noise_mask, 'trId'].astype(str)

        return df

    def _generate_candidates(self, df: pd.DataFrame, training: bool = False):
        """
        Generates pairs (i, j) by sorting by Amount.
        This brings recurring transactions (which usually have identical amounts) close together.
        """
        # Sort by Account -> Amount -> Date
        # This groups similar amounts together within the same account.
        df = df.sort_values(['accountId', 'amount', 'date']).reset_index(drop=True)

        features = []
        labels = []
        pair_indices = []

        # Convert columns to numpy for speed
        acc_ids = df['accountId'].values
        amounts = df['amount'].values.astype(float)
        dates = df['date'].values
        texts = df['bankRawDescription'].values

        pids = None
        if training:
            pids = df['patternId'].values

        n = len(df)
        window = self.config.window_size

        logger.info(f"Generating pairs for {n} transactions (Window={window})...")

        for i in range(n):
            # Sliding window: compare i with i+1 ... i+window
            for offset in range(1, window + 1):
                j = i + offset
                if j >= n: break

                # Hard Constraint 1: Must be same account
                if acc_ids[i] != acc_ids[j]:
                    break  # Sorted by account, so we can stop scanning this window

                # Hard Constraint 2: Amount must be somewhat close?
                # Actually, since we sorted by amount, they ARE close.
                # But let's calculate the actual diff.
                a1, a2 = amounts[i], amounts[j]
                diff_amt = abs(a1 - a2)

                # Optimization: If amounts differ by huge % (e.g. 10.00 vs 1000.00),
                # stop scanning this window (since list is sorted).
                if diff_amt > 10.0 and abs(a1) < 100:  # Heuristic
                    break

                # --- Feature 1: String Distance ---
                s1, s2 = texts[i], texts[j]
                str_score = fuzz.token_sort_ratio(s1, s2)

                if str_score < self.config.string_threshold:
                    continue

                # --- Feature 2: Amount Features ---
                max_amt = max(abs(a1), abs(a2)) + 1e-9
                rel_diff_amt = diff_amt / max_amt
                is_exact_amt = 1 if diff_amt < 0.01 else 0

                # --- Feature 3: Date Delta ---
                # Convert nanoseconds to days
                d1, d2 = dates[i], dates[j]
                diff_days = abs((d1 - d2).astype('timedelta64[D]').astype(int))

                # Feature Vector
                # We treat 'diff_days' as a raw number. The tree model will learn
                # splits like "if diff_days is approx 7 or 14 or 30..."
                features.append([
                    str_score,
                    diff_amt,
                    rel_diff_amt,
                    is_exact_amt,
                    diff_days
                ])

                # Keep track of original indices to reconstruct graph later
                pair_indices.append((i, j))

                if training:
                    # 1 if Same Pattern, 0 otherwise
                    is_match = 1 if (pids[i] == pids[j]) else 0
                    labels.append(is_match)

        logger.info(f"Generated {len(features)} pairs.")

        return (
            np.array(features, dtype=np.float32),
            np.array(labels, dtype=np.int32) if training else None,
            pair_indices,
            df
        )

    def fit(self, df: pd.DataFrame):
        df_clean = self._preprocess(df, training=True)
        X, y, _, _ = self._generate_candidates(df_clean, training=True)

        if len(X) == 0:
            print("No valid pairs found. Check data.")
            return

        print(f"Training on {len(X)} pairs...")
        # Train/Val Split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)

        # Validate
        y_prob = self.model.predict_proba(X_val)[:, 1]
        ap = average_precision_score(y_val, y_prob)
        print(f"--- Validation Results ---")
        print(f"Average Precision (PR-AUC): {ap:.4f}")

        # Feature Importance (Permutation importance is better, but this is quick proxy)
        print("Model trained.")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = self._preprocess(df, training=False)
        X, _, pair_indices, df_sorted = self._generate_candidates(df_clean, training=False)

        if len(X) == 0:
            return df_sorted

        # 1. Predict Link Probabilities
        probs = self.model.predict_proba(X)[:, 1]

        # 2. Build Adjacency Matrix
        # Only keep edges with high confidence
        mask = probs > self.config.match_threshold
        valid_pairs = np.array(pair_indices)[mask]

        n_nodes = len(df_sorted)
        if len(valid_pairs) == 0:
            df_sorted['pred_group_id'] = -1
            return df_sorted

        # Construct Graph
        row = valid_pairs[:, 0]
        col = valid_pairs[:, 1]
        data = np.ones(len(valid_pairs))

        adj_matrix = csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))

        # 3. Connected Components (Clustering)
        n_components, labels = connected_components(
            csgraph=adj_matrix,
            directed=False,
            return_labels=True
        )

        df_sorted['pred_group_id'] = labels

        # 4. Post-Process: Identify Cycles
        # Filter out groups of size 1 (noise)
        group_counts = df_sorted['pred_group_id'].value_counts()
        valid_groups = group_counts[group_counts > 1].index

        # Mark noise
        df_sorted.loc[~df_sorted['pred_group_id'].isin(valid_groups), 'pred_group_id'] = -1

        return df_sorted


def load_data(random_state:int|None=None, downsample:float|None=None):
    conf = MultiExpConfig()
    if random_state is not None:
        conf.random_state = random_state
    if downsample is not None:
        conf.downsample = downsample

    return load_data_for_config(conf)


if __name__ == "__main__":
    # Load your uploaded file
    print("Loading data...")

    train_df, val_df, test_df = load_data(random_state=0x5EED, downsample=0.1)

    # Instantiate and Train
    model = PairwiseRecurrenceModel()
    model.fit(train_df)

    # Predict (Simulating inference on the same data for demo)
    print("\nRunning Inference...")
    results = model.predict(val_df)

    # Display a few found patterns
    recurring = results[results['pred_group_id'] != -1]
    print(f"\nFound {len(recurring)} recurring transactions out of {len(val_df)} total.")

    # Show 5 random groups
    sample_groups = recurring['pred_group_id'].unique()[:5]
    for gid in sample_groups:
        print(f"\n--- Group {gid} ---")
        group = recurring[recurring['pred_group_id'] == gid].sort_values('date')
        print(group[['date', 'amount', 'bankRawDescription']].to_string(index=False))