from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from common.config import FieldConfig
from common.feature_processor import FeatProcParams
from tft.tft_data_clustered import prepare_clustered_tft_data, build_clustered_tft_dataset
from tft.tft_runner import TFTRunner


@pytest.fixture
def mock_embedding_service():
    """Mocks the heavy EmbeddingService to return random vectors instantly."""
    service = MagicMock()
    # Return a random (Batch, 768) array when embed() is called
    service.embed.side_effect = lambda texts: np.random.rand(len(texts), 768).astype(np.float32)
    return service


@pytest.fixture
def sample_data(field_config):
    """Creates a minimal DataFrame for TFT testing."""
    # Create 2 accounts, each with 10 transactions
    data = []
    for acc_id in ["A1", "B2"]:
        for i in range(10):
            data.append({
                field_config.accountId: acc_id,
                field_config.date: pd.Timestamp("2023-01-01") + pd.Timedelta(days=i),
                field_config.amount: -100.0 if i % 2 == 0 else -50.0,  # Alternating amounts to create clusters
                field_config.text: f"Txn {i}",
                field_config.label: 1 if i > 5 else 0,
                field_config.trId: f"{acc_id}_{i}"
            })
    return pd.DataFrame(data)


def test_tft_save_load_pipeline(sample_data, mock_embedding_service, tmp_path):
    """
    End-to-End System Test:
    1. Prepares data (clustering + pca).
    2. Builds TFT dataset.
    3. Trains a model for 1 epoch.
    4. Saves it to a temp path.
    5. Loads it back from the temp path.
    """
    field_config = FieldConfig()

    # 1. Setup Paths
    # use pytest's tmp_path fixture for automatic cleanup
    model_save_path = tmp_path / "test_model.pt"

    # 2. Prepare Data
    # We use minimal feature params to speed things up
    feat_params = FeatProcParams(
        use_cyclical_dates=True,
        use_continuous_amount=True,
        use_categorical_amount=False
    )

    # Run the clustered data pipeline
    df_prepped, _, _, meta = prepare_clustered_tft_data(
        df=sample_data,
        field_config=field_config,
        feat_params=feat_params,
        embedding_service=mock_embedding_service,
        fit_processor=True
    )

    # 3. Build Dataset & Loaders
    # Short encoder length because our dummy history is short (10 steps)
    dataset = build_clustered_tft_dataset(
        df_prepped,
        field_config,
        meta,
        max_encoder_length=5
    )

    # Split is artificial here; just use same dataset for train/val to ensure it runs
    dataloader = dataset.to_dataloader(train=True, batch_size=4)

    # 4. Initialize Runner
    runner = TFTRunner(
        train_ds=dataset,
        train_loader=dataloader,
        val_loader=dataloader,  # Mock validation
        val_df=df_prepped,
        max_epochs=1
    )

    # 5. Define Hyperparams (Minimal config)
    params = {
        "learning_rate": 1e-3,
        "hidden_size": 16,  # Small size for speed
        "attention_head_size": 1,
        "dropout": 0.1,
        "hidden_continuous_size": 8,
        "output_size": 2,
        "gradient_clip_val": 0.1
    }

    # --- PHASE 1: TRAIN & SAVE ---
    print("\n>>> Phase 1: Training...")
    assert not model_save_path.exists()

    trainer_1, model_1 = runner.train_single(params, model_path=str(model_save_path))

    assert model_save_path.exists(), "Model file was not created!"

    # Verify payload structure
    payload = torch.load(model_save_path)
    assert "state_dict" in payload
    assert "hyper_parameters" in payload
    assert payload["hyper_parameters"]["hidden_size"] == 16

    # --- PHASE 2: LOAD ---
    print("\n>>> Phase 2: Loading...")

    # We pass different params to ensure the loader ignores them and uses the saved ones
    wrong_params = params.copy()
    wrong_params["hidden_size"] = 999

    trainer_2, model_2 = runner.train_single(wrong_params, model_path=str(model_save_path))

    # Assertions
    assert model_2.hparams.hidden_size == 16, "Model did not load config from file!"
    assert model_2.hparams.hidden_size != 999

    # Check if weights match
    state_1 = model_1.state_dict()
    state_2 = model_2.state_dict()

    for key in state_1:
        assert torch.equal(state_1[key], state_2[key]), f"Weight mismatch for {key}"

    print("\n>>> Test Passed: Model trained, saved, and loaded successfully.")