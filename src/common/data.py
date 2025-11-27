import logging
from typing import Self
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
from common.config import FieldConfig
from multi.config import MultiFieldConfig

logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """Custom Dataset for text data."""
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return str(self.texts[idx]) # Ensure text is string


from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class TrainingSample:
    """
    A dataclass to hold one sample for our hybrid model.
    'frozen=True' makes it immutable, which is good practice for data samples.
    """
    x_text: torch.Tensor
    x_continuous: torch.Tensor
    x_categorical: torch.Tensor
    y: torch.Tensor

    @classmethod
    def collate_fn(cls, batch: list[Self]) -> Self:
        """
        Custom collate function to batch a list of TrainingSample dataclasses.
        This tells the DataLoader how to combine the individual samples.
        """

        # Use torch.stack to combine the tensors from each sample.
        # dim=0 creates the batch dimension.
        batched_x_text = torch.stack([s.x_text for s in batch], dim=0)
        batched_x_continuous = torch.stack([s.x_continuous for s in batch], dim=0)
        batched_x_categorical = torch.stack([s.x_categorical for s in batch], dim=0)
        batched_y = torch.stack([s.y for s in batch], dim=0)

        # Return a single TrainingSample containing the batched data
        return TrainingSample(
            x_text=batched_x_text,
            x_continuous=batched_x_continuous,
            x_categorical=batched_x_categorical,
            y=batched_y
        )


@dataclass(frozen=True)
class FeatureSet:
    """
    Holds the **complete** set of feature arrays for a dataset (e.g., train or test).
    """
    X_text: np.ndarray
    X_continuous: np.ndarray
    X_categorical: np.ndarray
    y: np.ndarray

class TransactionDataset(Dataset):
    """
    Custom PyTorch Dataset, now initialized from a FeatureSet dataclass.
    """

    def __init__(self, features: FeatureSet):

        # Convert to tensors from the FeatureSet
        # self.X_text = torch.tensor(features.X_text, dtype=torch.float32)
        # self.X_continuous = torch.tensor(features.X_continuous, dtype=torch.float32)
        # self.X_categorical = torch.tensor(features.X_categorical, dtype=torch.int64)
        # self.y = torch.tensor(features.y, dtype=torch.float32)
        self.X_text = torch.from_numpy(features.X_text).float()
        self.X_continuous = torch.from_numpy(features.X_continuous).float()
        self.X_categorical = torch.from_numpy(features.X_categorical).long()
        self.y = torch.from_numpy(features.y).float()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> TrainingSample:
        """
        (Unchanged from last time)
        Fetches one sample and returns it as a TrainingSample dataclass instance.
        """
        return TrainingSample(
            x_text=self.X_text[idx],
            x_continuous=self.X_continuous[idx],
            x_categorical=self.X_categorical[idx],
            y=self.y[idx]
        )

def create_mock_data(random_state:int, field_config: FieldConfig = FieldConfig(), n_samples: int = 2000) -> pd.DataFrame:
    """Creates a realistic, *balanced* dummy DataFrame for testing."""
    logger.info(f"Creating mock data ({n_samples} samples)...")

    np.random.seed(random_state)
    data = []

    for i in range(n_samples):
        record_type = np.random.choice(
            ['payroll', 'subscription', 'fee', 'coffee', 'random_large'],
            p=[0.15, 0.15, 0.05, 0.35, 0.30]
        )

        if record_type == 'payroll':
            amount = np.random.uniform(2000, 3000)
            base_text = np.random.choice(['PAYCHECK ABCORP', 'Direct Deposit - ABC', 'PAYROLL ABC INC'])
            day = 15 if i % 2 == 0 else 1
            for j in range(np.random.randint(3, 5)):
                month = 10 - j
                if month < 1:
                    month += 12
                date = f"{month:02d}/{day:02d}/2027 00:00:00"
                data.append({
                    field_config.date: date,
                    field_config.amount: round(amount + np.random.uniform(-1, 1), 2),
                    field_config.text: base_text,
                    field_config.label: 1
                })

        elif record_type == 'subscription':
            amount = np.random.choice([-15.99, -5.99, -10.00])
            base_text = np.random.choice(['NETFLIX.COM', 'AMZN-SUB', 'Spotify AB'])
            day = np.random.randint(1, 29)
            for j in range(np.random.randint(3, 5)):
                month = 10 - j
                if month < 1:
                    month += 12
                date = f"{month:02d}/{day:02d}/2027 00:00:00"
                data.append({
                    field_config.date: date,
                    field_config.amount: round(amount, 2),
                    field_config.text: f"{base_text} {np.random.randint(100, 900)}",
                    field_config.label: 1
                })

        elif record_type == 'fee':
            amount = -120.00
            text = "ANNUAL FEE - BANK"
            date = f"01/01/{2027 - np.random.randint(1, 4)} 00:00:00"
            data.append({
                field_config.date: date,
                field_config.amount: amount,
                field_config.text: text,
                field_config.label: 1
            })

        elif record_type == 'coffee':
            amount = np.random.uniform(-3, -12)
            text = np.random.choice(['Starbucks 8021', 'Random Cafe', 'Blue Bottle'])
            month = np.random.randint(8, 11)
            day = np.random.randint(1, 29)
            date = f"{month:02d}/{day:02d}/2027 00:00:00"
            data.append({
                field_config.date: date,
                field_config.amount: round(amount, 2),
                field_config.text: text,
                field_config.label: 0
            })

        elif record_type == 'random_large':
            amount = np.random.uniform(-50, -500)
            text = np.random.choice(['Amazon.com', 'One-Off Purchase', 'Best Buy'])
            month = np.random.randint(8, 11)
            day = np.random.randint(1, 29)
            date = f"{month:02d}/{day:02d}/2027 00:00:00"
            data.append({
                field_config.date: date,
                field_config.amount: round(amount, 2),
                field_config.text: text,
                field_config.label: 0
            })

    return pd.DataFrame(data).drop_duplicates()


def create_mock_account_data(field_config: FieldConfig) -> pd.DataFrame:
    """Creates a mock DataFrame for ONE account with clear patterns."""
    data = []

    # Group 1: Bi-weekly Paycheck
    for i in range(4):
        day = 1 if i % 2 == 0 else 15
        month = 10 - (i // 2)
        data.append({
            field_config.date: f"{month:02d}/{day:02d}/2027 00:00:00",
            field_config.amount: 2500.00 + np.random.uniform(-0.5, 0.5),
            field_config.text: "PAYCHECK ABCORP",
            field_config.label: "paycheck"
        })

    # Group 2: Monthly Subscription
    for i in range(3):
        day = 5
        month = 10 - i
        data.append({
            field_config.date: f"{month:02d}/{day:02d}/2027 00:00:00",
            field_config.amount: -15.99,
            field_config.text: "NETFLIX.COM 1234",
            field_config.label: "subscription"
        })

    # Group 3: Random Coffee
    for i in range(5):
        day = np.random.randint(1, 28)
        month = np.random.randint(8, 11)
        data.append({
            field_config.date: f"{month:02d}/{day:02d}/2027 00:00:00",
            field_config.amount: -1 * np.random.uniform(4.0, 8.0),
            field_config.text: "Starbucks",
            field_config.label: "coffee"
        })

    # Group 4: Similar-text, different amount/time
    data.append({
        field_config.date: "08/10/2027 00:00:00",
        field_config.amount: -100.00,
        field_config.text: "Amazon Purchase",
        field_config.label: "one-off"
    })
    data.append({
        field_config.date: "09/15/2027 00:00:00",
        field_config.amount: -50.00,
        field_config.text: "Amazon Mktplce",
        field_config.label: "one-off"
    })

    return pd.DataFrame(data)


def create_train_val_test_split(
    test_size: float,
    val_size: float,
    full_df: pd.DataFrame,
    random_state: int,
    field_config: FieldConfig|MultiFieldConfig = FieldConfig()
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info(f"Splitting {len(full_df)} rows into Train/Val/Test...")

    # --- 1. Split off the Holdout Test set (e.g., 20%) ---
    gss_test = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_val_idx, test_idx = next(gss_test.split(
        full_df, y=full_df[field_config.label], groups=full_df[field_config.accountId]
    ))

    df_train_val = full_df.iloc[train_val_idx]
    df_test_holdout = full_df.iloc[test_idx]

    # --- 2. Split the Train/Val set into Train and Validation ---
    # We need to adjust the val_size relative to the remaining data
    # e.g., if val_size=0.2 and test_size=0.2, we want 20% of original,
    # which is 0.2 / (1.0 - 0.2) = 0.25 of the remaining df_train_val.

    relative_val_size = val_size / (1.0 - test_size)

    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=relative_val_size, random_state=random_state
    )
    train_idx, val_idx = next(gss_val.split(
        df_train_val, y=df_train_val[field_config.label], groups=df_train_val[field_config.accountId]
    ))

    df_train = df_train_val.iloc[train_idx]
    df_val = df_train_val.iloc[val_idx]

    # --- 3. Sanity Checks ---
    train_acc = set(df_train[field_config.accountId].unique())
    val_acc = set(df_val[field_config.accountId].unique())
    test_acc = set(df_test_holdout[field_config.accountId].unique())

    assert train_acc.isdisjoint(val_acc), "Leakage: Train/Val account overlap"
    assert train_acc.isdisjoint(test_acc), "Leakage: Train/Test account overlap"
    assert val_acc.isdisjoint(test_acc), "Leakage: Val/Test account overlap"

    logger.info("Split complete. No account overlap.")
    logger.info(f"  Train:   {len(df_train)} rows, {len(train_acc)} accounts")
    logger.info(f"  Val:     {len(df_val)} rows, {len(val_acc)} accounts")
    logger.info(f"  Test:    {len(df_test_holdout)} rows, {len(test_acc)} accounts")

    return df_train, df_val, df_test_holdout


# ---------------------------------------------------------
# Filter Overlapping Bank Variations
# ---------------------------------------------------------
# Logic: If we have "Bank1" and "Bank2", we pick the one with
# the most rows (largest sample size), drop the others

def filter_unique_bank_variants(df: pd.DataFrame, field_config=FieldConfig()) -> pd.DataFrame:
    bank_name = field_config.bank_name
    if bank_name not in df.columns:
        return df

    print(f"Original row count: {len(df)}")

    # 1. Calculate row counts for each specific bank name
    bank_stats = df['bank_name'].value_counts().reset_index()
    bank_stats.columns = ['bank_name', 'row_count']

    # 2. Derive base name (remove digits) to find groups
    bank_stats['base_name'] = bank_stats['bank_name'].str.replace(r'\d+', '', regex=True)

    # 3. Sort by row_count descending so the largest is first
    bank_stats = bank_stats.sort_values('row_count', ascending=False)

    # 4. Select the largest bank_name for each base_name group
    # drop_duplicates keeps the first occurrence (which is now the largest due to sort)
    best_variants = bank_stats.drop_duplicates(subset='base_name', keep='first')
    allowed_banks = set(best_variants['bank_name'])

    print(
        f"Consolidating {len(bank_stats)} raw bank names into {len(allowed_banks)} unique base entities based on max row count.")

    # 5. Filter the main DataFrame
    filtered_df = df[df['bank_name'].isin(allowed_banks)].copy()


    print(f"Filtered row count: {len(filtered_df)} (Dropped {len(df) - len(filtered_df)} overlapping rows)")
    return filtered_df


def clean_text(series, hide_digits=False, trunc_repeating=True, lower=True):
    """
    Applies vectorized text cleaning operations on a pandas Series.
    """
    # Base clean: Convert to string and strip whitespace
    series = series.astype(str).str.strip()

    if lower:
        series = series.str.lower()

    if hide_digits:
        # Replace sequences of 3 or more digits with '###'
        series = series.str.replace(r'\d{3,}', '###', regex=True)

    if trunc_repeating:
        # Replace sequences of 3 or more IDENTICAL characters with just 3 of them
        series = series.str.replace(r'(.)\1{2,}', r'\1\1\1', regex=True)

    return series


def deduplicate(df: pd.DataFrame, field_config: FieldConfig=FieldConfig()) -> pd.DataFrame:
    """
    Deduplicates the DataFrame based on the transaction ID column.
    Keeps the first occurrence and logs the number of dropped rows.
    """
    initial_count = len(df)

    df = filter_unique_bank_variants(df, field_config)

    # Drop duplicates based on the configured transaction ID column
    df_deduped = df.drop_duplicates(subset=[field_config.trId], keep='first').copy()

    dropped_count = initial_count - len(df_deduped)
    if dropped_count > 0:
        logger.info(f"Deduplicated transactions: Dropped {dropped_count} rows based on '{field_config.trId}'.")

    mask = (df['isForeignCurrency'] == False) & (df['status'] == 'Cleared')
    df_deduped = df_deduped[mask]

    return df_deduped