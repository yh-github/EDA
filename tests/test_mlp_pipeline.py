import pytest
import pandas as pd
import torch
from common.feature_processor import HybridFeatureProcessor, FeatProcParams, FeatureHyperParams
from glob.classifier import HybridModel
from common.data import FeatureSet, TransactionDataset
from common.config import EmbModel


# Reusing fixtures from conftest.py implicitly

def test_full_pipeline_dry_run(sample_df, field_config):
    """
    Simulates the entire flow:
    Raw DF -> Processor -> FeatureSet -> Dataset -> Model -> Forward Pass
    """

    # 1. Feature Processing
    # Minimal config for speed
    feat_params = FeatProcParams(
        use_cyclical_dates=True,
        use_continuous_amount=True,
        use_categorical_amount=False  # Disable vocab logic for simplicity
    )

    processor = HybridFeatureProcessor.create(feat_params, field_config)
    meta = processor.fit(sample_df)
    features_df = processor.transform(sample_df)

    # 2. Mock Text Embeddings (Skip actual BERT)
    # 5 samples, dim 16
    mock_text_emb = np.random.rand(len(sample_df), 16).astype(np.float32)

    # 3. Create FeatureSet
    # Extract continuous parts
    cont_cols = meta.continuous_scalable_cols + meta.cyclical_cols
    X_cont = features_df[cont_cols].values.astype(np.float32)
    X_cat = np.zeros((len(sample_df), 0), dtype=np.int64)  # No categoricals enabled

    feature_set = FeatureSet(
        X_text=mock_text_emb,
        X_continuous=X_cont,
        X_categorical=X_cat,
        y=sample_df[field_config.label].values.astype(np.float32)
    )

    # 4. Build Dataset
    dataset = TransactionDataset(feature_set)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    # 5. Build Model
    feat_hyper_params = FeatureHyperParams.build(feature_set, meta)
    mlp_params = HybridModel.MlpHyperParams(mlp_hidden_layers=[8], dropout_rate=0.1)

    model = HybridModel(feature_config=feat_hyper_params, mlp_config=mlp_params)

    # 6. Run Forward Pass (Simulation of Training Step)
    batch = next(iter(loader))
    logits = model(batch.x_text, batch.x_continuous, batch.x_categorical)

    assert logits.shape == (2,)  # Batch size 2
    assert logits.requires_grad  # Ensure gradient graph is built