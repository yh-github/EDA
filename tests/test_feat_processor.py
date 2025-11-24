import pytest
import pandas as pd
import numpy as np
from common.feature_processor import HybridFeatureProcessor, FeatProcParams


def test_initialization(field_config):
    params = FeatProcParams.all_off()
    processor = HybridFeatureProcessor.create(params, field_config)
    assert not processor.already_fitted
    assert processor.k_top == 0  # Should default to 0 if all_off


def test_fit_builds_vocab(sample_df, field_config):
    """Ensure fit creates correct vocab for categorical amounts."""
    params = FeatProcParams(use_categorical_amount=True, k_top=1, n_bins=2)
    processor = HybridFeatureProcessor.create(params, field_config)

    meta = processor.fit(sample_df)

    assert processor.already_fitted
    assert 'amount_token_id' in meta.categorical_features
    # Top K=1 should be -100.0 (appears twice)
    assert -100.0 in processor.top_k_amounts
    assert '[UNKNOWN]' in processor.vocab_map


def test_transform_dates(sample_df, field_config):
    params = FeatProcParams(use_cyclical_dates=True, use_categorical_dates=True)
    processor = HybridFeatureProcessor.create(params, field_config)

    # Transform doesn't require fit for pure math features (dates),
    # but logically we usually fit first.
    processor.fit(sample_df)
    res = processor.transform(sample_df)

    assert 'day_of_week_sin' in res.columns
    assert 'day_of_week_id' in res.columns

    # Check values for '2023-01-01' (Sunday = 6)
    idx_0 = res.iloc[0]
    # sin(6 * 2pi / 7) is approx -0.78
    assert np.isclose(idx_0['day_of_week_sin'], np.sin(6 * 2 * np.pi / 7))
    assert idx_0['day_of_week_id'] == 6


def test_transform_behavioral(sample_df, field_config):
    params = FeatProcParams(use_behavioral_features=True)
    processor = HybridFeatureProcessor.create(params, field_config)
    processor.fit(sample_df)
    res = processor.transform(sample_df)

    assert 'acc_stat_amount_mean' in res.columns

    # A1 has amounts [-100, -100, -50] -> log(abs(x)+1) -> [4.615, 4.615, 3.931]
    # Mean approx 4.387
    expected_mean = np.mean(np.log(np.abs([-100, -100, -50]) + 1))
    assert np.isclose(res.iloc[0]['acc_stat_amount_mean'], expected_mean, atol=1e-3)


def test_tokenize_amount_logic(sample_df, field_config):
    params = FeatProcParams(use_categorical_amount=True, k_top=1, n_bins=2)
    processor = HybridFeatureProcessor.create(params, field_config)
    processor.fit(sample_df)

    # -100 is top 1
    token_100 = processor._tokenize_amount(-100.0)
    # -50 is not top 1
    token_50 = processor._tokenize_amount(-50.0)

    assert token_100 != token_50
    assert token_100 == processor.vocab_map[-100.0]


def test_processor_raises_if_not_fitted(sample_df, field_config):
    params = FeatProcParams(use_categorical_amount=True)
    processor = HybridFeatureProcessor.create(params, field_config)

    with pytest.raises(RuntimeError):
        processor.transform(sample_df)


def test_empty_dataframe(field_config):
    """Processor should handle empty DFs gracefully or raise specific error."""
    df = pd.DataFrame(columns=[field_config.accountId, field_config.date, field_config.amount])
    params = FeatProcParams(use_continuous_amount=True)
    processor = HybridFeatureProcessor.create(params, field_config)

    # Should run without error
    meta = processor.fit(df)
    res = processor.transform(df)
    assert len(res) == 0