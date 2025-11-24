import pandas as pd
import pytest

from common.config import FieldConfig
from common.feature_processor import FeatProcParams


@pytest.fixture
def field_config() -> FieldConfig:
    return FieldConfig(
        date='trx_date',
        amount='trx_amount',
        text='trx_desc',
        label='is_recurring',
        accountId='acc_id',
        trId='trx_id'
    )

@pytest.fixture
def sample_df(field_config: FieldConfig) -> pd.DataFrame:
    """Creates a small, deterministic DataFrame for testing."""
    data = {
        field_config.accountId: ['A1', 'A1', 'A1', 'B2', 'B2'],
        field_config.date: [
            '2023-01-01 10:00:00',
            '2023-01-15 10:00:00',
            '2023-02-01 10:00:00',
            '2023-03-01 09:00:00',
            '2023-03-02 09:00:00'
        ],
        field_config.amount: [-100.0, -100.0, -50.0, -10.50, 2000.0],
        field_config.text: ['Netflix', 'Netflix', 'Amazon', 'Coffee', 'Paycheck'],
        field_config.label: [1, 1, 0, 0, 1],
        field_config.trId: ['t1', 't2', 't3', 't4', 't5']
    }
    return pd.DataFrame(data)

@pytest.fixture
def full_feat_params() -> FeatProcParams:
    return FeatProcParams(
        use_cyclical_dates=True,
        use_categorical_dates=True,
        use_continuous_amount=True,
        use_categorical_amount=True,
        use_is_positive=True,
        use_behavioral_features=True,
        k_top=2,
        n_bins=2
    )