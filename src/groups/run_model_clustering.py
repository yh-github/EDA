import logging
import argparse
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.metrics import classification_report

# Suppress Warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*n_jobs.*")

from common.config import ExperimentConfig, FieldConfig, EmbModel
from common.feature_processor import FeatProcParams
from common.embedder import EmbeddingService
from common.log_utils import setup_logging
from pointwise.runner import ExpRunner
from pointwise.classifier import HybridModel
from groups.model_clusterer import ModelBasedClusterer
from common.data import FeatureSet

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/rec_data2.csv")
    parser.add_argument("--output_path", type=str, default="results/model_clustering.csv")
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def run_experiment():
    args = parse_args()
    setup_logging(Path("logs/"), "model_clustering")

    # 1. Load Data
    df = pd.read_csv(args.data_path)
    field_config = FieldConfig()
    df_clean = df.dropna(subset=[field_config.date, field_config.amount, field_config.text, field_config.label])

    # 2. Configure
    exp_config = ExperimentConfig(epochs=5, batch_size=128)
    feat_params = FeatProcParams(use_cyclical_dates=True, use_continuous_amount=True, use_categorical_amount=False)
    emb_params = EmbeddingService.Params(model_name=EmbModel.MPNET)
    mlp_params = HybridModel.MlpHyperParams(mlp_hidden_layers=[128, 64], dropout_rate=0.25, text_projection_dim=64)

    runner = ExpRunner.create(exp_config, df_clean, emb_params, feat_params, mlp_params, field_config)

    # 3. Split & Build (3-Way)
    df_train, df_val, df_test = runner.create_train_val_test_split(test_size=0.2, val_size=0.2)
    logger.info(">>> STEP 1: Training Pointwise Model (Train/Val)...")

    train_fs, val_fs, test_fs, processor, meta = runner.build_data_three_way(df_train, df_val, df_test)

    # 4. Train
    val_metrics, model = runner.run_experiment_and_return_model(train_fs, val_fs, meta)
    logger.info(f"Pointwise Validation Metrics (Best Epoch): {val_metrics}")

    # --- NEW: Evaluate Pointwise on Test Set ---
    logger.info(">>> STEP 1.5: Evaluating Pointwise Model on Test Set...")
    test_metrics = runner.evaluate_model_on_set(model, test_fs)
    logger.info(
        f"Pointwise TEST Metrics: F1={test_metrics['f1']}, P={test_metrics['precision']}, R={test_metrics['recall']}, AUC={test_metrics['roc_auc']}")

    # 5. Cluster (Test Set Only)
    logger.info(">>> STEP 2: Running Model-Based Clustering on Held-Out Test Set...")
    clusterer = ModelBasedClusterer(model, processor, min_samples=2, use_gpu=args.gpu, voting_threshold=0.5)

    results = []
    test_indices = df_test.groupby(field_config.accountId).indices

    for acc_id, idxs in test_indices.items():
        indices = np.sort(idxs)

        sub_fs = FeatureSet(
            X_text=test_fs.X_text[indices],
            X_continuous=test_fs.X_continuous[indices],
            X_categorical=test_fs.X_categorical[indices],
            y=test_fs.y[indices]
        )

        sub_df = df_test.iloc[indices].copy()
        res = clusterer.cluster_features(sub_fs, sub_df)
        results.append(res.clustered_df)

    # 6. Evaluate
    final_df = pd.concat(results)
    y_true = final_df[field_config.label].astype(int)
    y_pred = final_df['prediction']

    print("\n" + classification_report(y_true, y_pred))
    final_df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    run_experiment()