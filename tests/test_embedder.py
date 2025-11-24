import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from common.embedder import EmbeddingService, EmbModel


@pytest.fixture
def mock_transformers():
    """Mocks HuggingFace transformers to prevent downloading models."""
    with patch("common.embedder.AutoTokenizer") as mock_tok, \
            patch("common.embedder.AutoModel") as mock_mod:
        # Setup Tokenizer Mock
        mock_tok_instance = MagicMock()
        mock_tok.from_pretrained.return_value = mock_tok_instance
        # When tokenizer is called, return dict of tensors
        mock_tok_instance.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]])
        }

        # Setup Model Mock
        mock_mod_instance = MagicMock()
        mock_mod.from_pretrained.return_value = mock_mod_instance
        # Mock forward pass output
        mock_output = MagicMock()
        # [Batch=1, Seq=2, Dim=10]
        mock_output.last_hidden_state = torch.rand(1, 2, 10)
        mock_output.pooler_output = torch.rand(1, 10)
        mock_mod_instance.return_value = mock_output
        mock_mod_instance.to.return_value = mock_mod_instance
        mock_mod_instance.eval.return_value = mock_mod_instance

        yield mock_tok, mock_mod


def test_embedder_caching(mock_transformers, tmp_path):
    """Test that embedding service computes once and then uses cache."""

    # Patch the cache dir to use a temporary pytest directory
    with patch("common.embedder.CACHE_DIR_BASE", str(tmp_path)):
        params = EmbeddingService.Params(model_name=EmbModel.MPNET, batch_size=2)
        service = EmbeddingService.create(params)

        texts = ["hello", "world"]

        # First call: Should compute (trigger model)
        emb1 = service.embed(texts)
        assert emb1.shape == (2, 10)  # Mock dim is 10

        # Check call counts
        # We process 2 texts in 1 batch (batch_size=2)
        assert mock_transformers[1].from_pretrained.call_count == 1
        model_instance = mock_transformers[1].from_pretrained.return_value
        assert model_instance.call_count == 1

        # Second call: Should hit memory cache (no model call)
        emb2 = service.embed(texts)
        assert np.array_equal(emb1, emb2)
        assert model_instance.call_count == 1  # Count should NOT increment


def test_embedder_empty_list(mock_transformers):
    params = EmbeddingService.Params(model_name=EmbModel.MPNET)
    service = EmbeddingService.create(params)
    res = service.embed([])
    assert len(res) == 0