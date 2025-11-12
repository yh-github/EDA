import logging
from typing import Self

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
from config import FieldConfig

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
        self.X_text = torch.tensor(features.X_text, dtype=torch.float32)
        self.X_continuous = torch.tensor(features.X_continuous, dtype=torch.float32)
        self.X_categorical = torch.tensor(features.X_categorical, dtype=torch.int64)
        self.y = torch.tensor(features.y, dtype=torch.float32)

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
