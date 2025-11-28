import os
import random
import logging
import pandas as pd
import numpy as np
from multi.config import MultiExpConfig, MultiFieldConfig
from common.data import clean_text

logger = logging.getLogger(__name__)


def mock_data_generator(num_accounts=100) -> pd.DataFrame:
    """Creates dummy data if no CSV is found."""
    logger.info("Generating synthetic data for testing...")
    data = []
    for acc_id in range(num_accounts):
        num_tx = random.randint(20, 150)
        for i in range(num_tx):
            is_rec = random.random() < 0.2
            cycle = random.choice(['monthly', 'onceAWeek']) if is_rec else 'None'
            pat_id = random.randint(1, 5) if is_rec else None
            amount = 15.99 if is_rec else random.uniform(5, 100)
            if random.random() > 0.5:
                amount *= -1

            data.append({
                'accountId': acc_id,
                'trId': f"{acc_id}_{i}",
                'bankRawDescription': f"PAYMENT TO {'NETFLIX' if is_rec else 'STORE'} {random.randint(1, 100)}",
                'counter_party': 'Netflix' if is_rec else '',
                'amount': amount,
                'date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=random.randint(0, 150)),
                'isRecurring': is_rec,
                'patternId': pat_id,
                'patternCycle': cycle
            })
    return pd.DataFrame(data)


def load_and_prepare_data(config: MultiExpConfig) -> pd.DataFrame:
    """
    Centralized data loading pipeline:
    1. Reads CSV or Mock data.
    2. Casts ID columns.
    3. Checks Feature Availability (Counter Party).
    4. Downsamples data (if configured).
    5. Cleans text.

    Args:
        config: The experiment configuration object. NOTE: This function may
                modify config.use_counter_party based on data availability.
    """
    field_config = MultiFieldConfig()

    # 1. Load Data
    if config.data_path.lower() == "mock":
        df = mock_data_generator()
    elif config.data_path and os.path.exists(config.data_path):
        logger.info(f"Loading data from {config.data_path}...")
        df = pd.read_csv(config.data_path, low_memory=False)
    else:
        raise FileNotFoundError(f"No CSV found at path={config.data_path}")

    logger.info(f"Loaded {len(df)} raw transactions.")

    # 2. Type Enforcement
    if field_config.accountId in df.columns:
        df[field_config.accountId] = df[field_config.accountId].astype(str)
    if field_config.trId in df.columns:
        df[field_config.trId] = df[field_config.trId].astype(str)

    # 3. Feature Availability Check (Counter Party)
    if field_config.counter_party in df.columns:
        valid_cp = df[field_config.counter_party].replace('', np.nan).notna()
        coverage = valid_cp.sum() / len(df)
        logger.info(f"Counter Party Coverage: {coverage:.2%}")

        if coverage < 0.5:
            logger.warning("Counter Party coverage low (<50%). Disabling feature in config.")
            config.use_counter_party = False
        else:
            # Only enable if it was originally requested
            if config.use_counter_party:
                logger.info("Counter Party coverage good. Keeping feature enabled.")
            else:
                logger.info("Counter Party coverage good, but feature disabled in config.")
    else:
        logger.warning("Counter Party column missing. Disabling feature in config.")
        config.use_counter_party = False

    # 4. Downsample (Account-based)
    if 0.0 < config.downsample < 1.0:
        logger.info(f"Downsampling to {config.downsample:.0%} of accounts...")
        account_ids = df[field_config.accountId].unique()
        rng = np.random.default_rng(config.random_state)

        n_select = max(1, int(len(account_ids) * config.downsample))
        selected_ids = rng.choice(account_ids, size=n_select, replace=False)

        df = df[df[field_config.accountId].isin(selected_ids)].copy()
        logger.info(f"Dataset size after downsampling: {len(df)} rows ({len(selected_ids)} accounts)")

    # 5. Clean Text
    logger.info("Cleaning text descriptions...")
    df[field_config.text] = clean_text(df[field_config.text])

    return df