import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, average_precision_score
from multi.config import MultiExpConfig, MultiFieldConfig
from multi.inference import MultiPredictor


def bcubed_precision_recall(true_ids, pred_ids):
    n = len(true_ids)
    if n == 0: return 0.0, 0.0

    def get_clusters(id_list):
        clusters = {}
        for idx, val in enumerate(id_list):
            if val is None or val == -1 or val == 'None' or pd.isna(val):
                key = f"NOISE_{idx}"
            else:
                key = val
            if key not in clusters: clusters[key] = set()
            clusters[key].add(idx)
        return clusters

    true_clusters = get_clusters(true_ids)
    pred_clusters = get_clusters(pred_ids)

    true_idx_map = {idx: k for k, s in true_clusters.items() for idx in s}
    pred_idx_map = {idx: k for k, s in pred_clusters.items() for idx in s}

    precision_sum = 0.0
    recall_sum = 0.0

    for i in range(n):
        true_cl = true_clusters[true_idx_map[i]]
        pred_cl = pred_clusters[pred_idx_map[i]]
        intersection = len(true_cl.intersection(pred_cl))

        precision_sum += intersection / len(pred_cl)
        recall_sum += intersection / len(true_cl)

    return precision_sum / n, recall_sum / n


def evaluate_run(data_path, model_path):
    # 1. Setup
    config = MultiExpConfig()
    fc = MultiFieldConfig()

    # 2. Predictor
    predictor = MultiPredictor(model_path, config)

    # 3. Predict
    print(f"Loading data from {data_path}...")
    print(f"Loading data from {data_path}...")
    if data_path.endswith('.pkl'):
        import pickle
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        if isinstance(data_dict, dict) and 'test' in data_dict:
            print("Detected cached split file. Loading 'test' set...")
            df = data_dict['test']
        else:
            # Fallback if it's just a dataframe pickled
            df = data_dict
    else:
        df = pd.read_csv(data_path)
    print("Running Inference...")
    pred_df = predictor.predict(df)
    if pred_df is None or pred_df.empty:
        print("Error: No predictions returned.")
        return

    # 4. Metrics
    print("\n--- Evaluation Results ---")

    # A. Level 1: Binary Detection
    y_true_bin = df[fc.label].fillna(False).astype(int)

    # Re-index pred_df to match input df order (handles any drops/sorts during inference)
    pred_df = pred_df.reindex(df.index)

    y_pred_bin = pred_df['pred_isRecurring'].fillna(False).astype(int)
    y_pred_prob = pred_df['pred_recurring_prob'].fillna(0.0)

    bin_f1 = f1_score(y_true_bin, y_pred_bin)
    bin_prauc = average_precision_score(y_true_bin, y_pred_prob)

    print(f"Level 1 - Detection F1:     {bin_f1:.4f}")
    print(f"Level 1 - Detection PR-AUC: {bin_prauc:.4f}")

    # B. Level 2: Grouping (B-Cubed)
    b_precisions = []
    b_recalls = []

    # Use original df grouping to ensure we catch all accounts
    for acc_id, group in df.groupby(fc.accountId):
        pred_group = pred_df.loc[group.index]
        p, r = bcubed_precision_recall(
            group[fc.patternId].tolist(),
            pred_group['pred_patternId'].tolist()
        )
        b_precisions.append(p)
        b_recalls.append(r)

    avg_bp = np.mean(b_precisions) if b_precisions else 0.0
    avg_br = np.mean(b_recalls) if b_recalls else 0.0
    b_f1 = 2 * (avg_bp * avg_br) / (avg_bp + avg_br + 1e-8)

    print(f"Level 2 - Clustering B-Cubed F1: {b_f1:.4f} (P: {avg_bp:.4f}, R: {avg_br:.4f})")

    # C. Level 3: Cycle Accuracy
    mask = (y_true_bin == 1) & (y_pred_bin == 1)
    if mask.sum() > 0:
        acc = accuracy_score(
            df.loc[mask, fc.patternCycle].astype(str),
            pred_df.loc[mask, 'pred_patternCycle'].astype(str)
        )
        print(f"Level 3 - Cycle Accuracy (TP):   {acc:.4f}")
    else:
        print("Level 3 - Cycle Accuracy: N/A")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    evaluate_run(args.data, args.model)