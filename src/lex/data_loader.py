import numpy as np
import pandas as pd
from multi.reload_utils import load_cached_data
from common.config import FieldConfig
import logging

logger = logging.getLogger(__name__)

DATA_PATH = 'data/all_data.csv'


def load_data(path=DATA_PATH):
    """Loads the dataset."""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path, dtype={'accountId': str})
    print(f"Loaded {len(df)} transactions.")
    return df


def preprocess_data(df):
    """
    Truncates each account to its most recent 150 days.
    Filters out accounts with < 35 days of data after truncation.
    """
    print("Preprocessing data...")

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # 1. Truncate to most recent 150 days per account
    # Find max date per account
    max_dates = df.groupby('accountId')['date'].transform('max')

    # Filter: date > (max_date - 150 days)
    # Note: 150 days window means [max_date - 150, max_date]
    cutoff_dates = max_dates - pd.Timedelta(days=150)
    df_truncated = df[df['date'] >= cutoff_dates].copy()

    print(f"Transactions after truncation: {len(df_truncated)}")

    # 2. Filter accounts with < 35 days
    account_stats = df_truncated.groupby('accountId')['date'].agg(['min', 'max'])
    account_stats['duration'] = (account_stats['max'] - account_stats['min']).dt.days

    print(f"Total accounts after truncation: {len(account_stats)}")

    valid_accounts = account_stats[account_stats['duration'] >= 35]
    print(f"Accounts with >= 35 days: {len(valid_accounts)}")

    # Filter dataframe
    df_final = df_truncated[df_truncated['accountId'].isin(valid_accounts.index)].copy()
    print(f"Transactions after filtering: {len(df_final)}")

    return df_final


def split_data(df, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Splits data into train/val/test by account ID.
    """
    print("Splitting data...")
    account_ids = df['accountId'].unique()
    np.random.seed(random_state)
    np.random.shuffle(account_ids)

    n = len(account_ids)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)

    test_ids = account_ids[:n_test]
    val_ids = account_ids[n_test:n_test + n_val]
    train_ids = account_ids[n_test + n_val:]

    print(f"Train accounts: {len(train_ids)}")
    print(f"Val accounts: {len(val_ids)}")
    print(f"Test accounts: {len(test_ids)}")

    train_df = df[df['accountId'].isin(train_ids)]
    val_df = df[df['accountId'].isin(val_ids)]
    test_df = df[df['accountId'].isin(test_ids)]

    return train_df, val_df, test_df






def preprocess_lex_features(df: pd.DataFrame, field_config: FieldConfig=FieldConfig()) -> pd.DataFrame:
    """
    Applies Lex-specific preprocessing (e.g., specific truncation logic)
    AFTER loading the canonical splits.
    """
    df = df.copy()

    # Ensure date format
    if not np.issubdtype(df[field_config.date].dtype, np.datetime64):
        df[field_config.date] = pd.to_datetime(df[field_config.date])


    return df


def load_lex_splits(random_state=0x5EED2, downsample=0.15):
    """
    Loads the exact Train/Val/Test splits used by the Multi model
    from the disk cache.
    """

    logger.info(f"Loading cached splits (Seed={random_state}, Downsample={downsample})...")

    train_df, val_df, test_df = load_cached_data(
        random_state=random_state,
        downsample=downsample
    )

    logger.info(f"Loaded - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return preprocess_lex_features(train_df), preprocess_lex_features(val_df), preprocess_lex_features(test_df)




if __name__ == "__main__":
    df = load_data()
    df_clean = preprocess_data(df)
    train_df, val_df, test_df = split_data(df_clean)

    # Save splits (optional, or just return them)
    # train_df.to_csv('train.csv', index=False)
    # val_df.to_csv('val.csv', index=False)
    # test_df.to_csv('test.csv', index=False)

    print("Data loading and splitting complete.")
