import logging
import pandas as pd
import argparse
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.model_selection import GroupShuffleSplit  # Added import

from common.config import FieldConfig, EmbModel, ExperimentConfig
# Removed create_train_val_test_split import to avoid confusion
from common.feature_processor import FeatProcParams
from common.embedder import EmbeddingService
from common.log_utils import setup_logging
from groups.emb_clusterer import EmbClusterer, GroupClassifier

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Supervised Clustering Pipeline")
    parser.add_argument("--data_path", type=str, default="data/rec_data2.csv")
    parser.add_argument("--output_path", type=str, default="results/supervised_clustering.csv")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU")
    return parser.parse_args()


def run_pipeline():
    args = parse_args()
    setup_logging(Path("logs/"), "supervised_clustering")

    # 1. Load Data
    logger.info("Loading Data...")
    df = pd.read_csv(args.data_path)
    field_config = FieldConfig()
    df_clean = df.dropna(subset=[field_config.date, field_config.amount, field_config.text, field_config.label]).copy()
    df_clean[field_config.label] = df_clean[field_config.label].astype(int)

    exp_config = ExperimentConfig()

    # --- FIX: Direct 2-Way Split (Train/Test) ---
    logger.info(f"Splitting Data (Train/Test) with random_state={exp_config.random_state}...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=exp_config.random_state)

    # Split by Account ID to prevent leakage
    train_idx, test_idx = next(gss.split(df_clean, groups=df_clean[field_config.accountId]))

    train_df = df_clean.iloc[train_idx].copy()
    test_df = df_clean.iloc[test_idx].copy()

    logger.info(f"Train: {len(train_df)} rows ({train_df[field_config.accountId].nunique()} accounts)")
    logger.info(f"Test:  {len(test_df)} rows ({test_df[field_config.accountId].nunique()} accounts)")

    # 2. Setup Components
    emb_service = EmbeddingService.create(EmbeddingService.Params(model_name=EmbModel.MPNET, batch_size=512))

    feat_params = FeatProcParams(
        use_continuous_amount=True,
        use_cyclical_dates=True,
        use_categorical_dates=False,
        use_categorical_amount=False
    )

    # Relaxed Clusterer for High Recall (Candidate Generation)
    clusterer = EmbClusterer(
        field_config, feat_params, emb_service,
        min_samples=2,  # Catch everything (pairs)
        cluster_epsilon=0.0,  # Tight clusters
        use_gpu=args.gpu
    )

    classifier = GroupClassifier()

    # 3. TRAINING PHASE
    logger.info("--- TRAINING PHASE ---")
    train_candidates = []

    # Iterate Train Accounts
    # GroupBy is safe here because train_df contains only training accounts
    for _, acc_df in train_df.groupby(field_config.accountId):
        try:
            _, cands = clusterer.extract_candidates(acc_df)
            train_candidates.extend(cands)
        except Exception:
            pass

    logger.info(f"Generated {len(train_candidates)} candidates from training set.")

    # Train Classifier
    classifier.fit(train_candidates)

    # 4. TESTING PHASE
    logger.info("--- TESTING PHASE ---")
    results = []

    for _, acc_df in test_df.groupby(field_config.accountId):
        try:
            # 1. Generate Candidates
            df_clustered, candidates = clusterer.extract_candidates(acc_df)
            df_clustered['prediction'] = 0  # Default to 0

            # 2. Filter using Trained Classifier
            for cand in candidates:
                is_recurring = classifier.predict(cand)

                if is_recurring:
                    # Apply label '1' to rows in this cluster
                    cluster_id = int(cand.group_id.split("_")[1])
                    df_clustered.loc[df_clustered['cluster_label'] == cluster_id, 'prediction'] = 1

            results.append(df_clustered)

        except Exception as e:
            logger.error(f"Error in test: {e}")
            fallback = acc_df.copy()
            fallback['prediction'] = 0
            results.append(fallback)

    # 5. Evaluation
    final_df = pd.concat(results, ignore_index=True)
    y_true = final_df[field_config.label].values
    y_pred = final_df['prediction'].values

    logger.info("\n" + "=" * 50)
    logger.info("FINAL TEST RESULTS")
    logger.info("=" * 50)

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    logger.info(f"Precision : {p:.4f}")
    logger.info(f"Recall    : {r:.4f}")
    logger.info(f"F1 Score  : {f1:.4f}")

    print(classification_report(y_true, y_pred))

    # Save
    final_df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    run_pipeline()