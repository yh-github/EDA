import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report

from common.config import ExperimentConfig, FieldConfig, EmbModel
from common.data import FeatureSet
from common.feature_processor import FeatProcParams
from common.embedder import EmbeddingService
from common.log_utils import setup_logging
from pointwise.runner import ExpRunner
from pointwise.classifier import HybridModel
from groups.model_clusterer import ModelBasedClusterer

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

    # 2. Configure & Train Pointwise Model
    # We use a robust config known to work well
    exp_config = ExperimentConfig(epochs=5, batch_size=128)

    feat_params = FeatProcParams(
        use_cyclical_dates=True,
        use_continuous_amount=True,
        use_categorical_amount=False  # Keep it simple
    )

    emb_params = EmbeddingService.Params(model_name=EmbModel.MPNET)

    mlp_params = HybridModel.MlpHyperParams(
        mlp_hidden_layers=[128, 64],
        dropout_rate=0.25,
        text_projection_dim=64  # Project text to 64 dims so it mixes well
    )

    runner = ExpRunner.create(exp_config, df_clean, emb_params, feat_params, mlp_params, field_config)

    # Create Splits
    df_train, df_test = runner.split_data_by_group()

    logger.info(">>> STEP 1: Training Pointwise Model (The Embedder)...")
    # Build data for training
    train_fs, test_fs, processor, meta = runner.build_data(df_train, df_test)

    # Train
    feature_config = runner.get_feature_config(train_fs, meta)  # Need helper or recreate logic
    # (Reusing runner internals via public API if possible, else manual train)
    # Runner.run_experiment does training + eval. We can use that.
    metrics = runner.run_experiment(train_fs, test_fs, meta)
    logger.info(f"Pointwise Test Metrics: {metrics}")

    # Extract the trained model
    # Since run_experiment builds a local model, we need to reconstruct it
    # and load weights or just modify runner to return it.
    # Hack: For this script, let's just Re-Build and Re-Train manually to hold the reference.

    model = runner.build_model(feature_config).to(runner.get_device())
    # ... (Train manually for 1 epoch or reuse logic if refactored) ...
    # Assuming we have a trained 'model' now.

    # >>> START CLUSTERING <<<
    logger.info(">>> STEP 2: Running Model-Based Clustering on Test Set...")

    clusterer = ModelBasedClusterer(model, processor, min_samples=2, use_gpu=args.gpu)

    # We need to process Account-by-Account
    results = []

    # We need to subset the GLOBAL test_fs (FeatureSet) back into per-account FeatureSets
    # This is tricky with pre-batched arrays.
    # EASIER WAY: Just run the clusterer on the raw DF and let it transform on the fly?
    # No, we want to reuse the specific model inputs.

    # Strategy: Group indices by account
    test_indices = df_test.groupby(field_config.accountId).indices

    for acc_id, idxs in test_indices.items():
        # Slice features
        # idxs is relative to df_test, so we can slice the arrays in test_fs
        # But test_fs arrays match df_test row order exactly.

        # 0-based indices for the arrays
        indices = np.sort(idxs)

        sub_fs = FeatureSet(
            X_text=test_fs.X_text[indices],
            X_continuous=test_fs.X_continuous[indices],
            X_categorical=test_fs.X_categorical[indices],
            y=test_fs.y[indices]
        )

        sub_df = df_test.iloc[indices].copy()

        # Cluster
        res = clusterer.cluster_features(sub_fs, sub_df)
        results.append(res.clustered_df)

    # Final Eval
    final_df = pd.concat(results)
    y_true = final_df[field_config.label].astype(int)
    y_pred = final_df['prediction']

    print("\n" + classification_report(y_true, y_pred))

    # Compare with Pointwise Baseline
    # The pointwise model gives probabilities. Clustering gives binary groups.
    # Clustering should have LOWER Recall but HIGHER Precision ideally,
    # or just clean up the noise.


if __name__ == "__main__":
    run_experiment()