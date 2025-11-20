import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from config import FieldConfig, get_device
from log_utils import setup_logging

# --- Setup ---
setup_logging(Path('logs/'), "train_neural_ts")
logger = logging.getLogger(__name__)
DEVICE = get_device()


# --- 1. Dataset ---

class TimeSeriesDataset(Dataset):
    """
    Converts Account Transactions into dense daily signals.
    Input:  Timeline of ALL transactions (1.0 = transaction occurred)
    Target: Timeline of RECURRING transactions (1.0 = recurring event occurred)
    """

    def __init__(self, df: pd.DataFrame, field_config: FieldConfig, seq_len: int = 180):
        self.seq_len = seq_len
        self.samples = []
        self.fc = field_config

        self._build_samples(df)

    def _build_samples(self, df: pd.DataFrame):
        logger.info(f"Building Time Series samples (Seq Len: {self.seq_len})...")

        # Convert dates once
        df = df.copy()
        df['dt'] = pd.to_datetime(df[self.fc.date])

        grouped = df.groupby(self.fc.accountId)

        for acc_id, group in grouped:
            if len(group) < 5: continue

            # Normalize dates to start at day 0
            start_date = group['dt'].min().normalize()
            days = (group['dt'] - start_date).dt.days.values

            # Filter out transactions beyond the sequence length
            mask_time = days < self.seq_len
            if not mask_time.any(): continue

            valid_days = days[mask_time]

            # --- Input Signal (All Activity) ---
            # Shape: [1, Seq_Len]
            input_tensor = torch.zeros((1, self.seq_len), dtype=torch.float32)
            # Mark days with activity as 1.0
            # (Advanced: Use normalized log-amounts instead of binary 1.0)
            input_tensor[0, valid_days] = 1.0

            # --- Target Signal (Only Recurring) ---
            target_tensor = torch.zeros((1, self.seq_len), dtype=torch.float32)

            # Identify recurring transactions
            rec_mask = (group[self.fc.label] == True).values & mask_time
            rec_days = days[rec_mask]

            if len(rec_days) > 0:
                # Mark recurring days as 1.0
                # We can use Gaussian smoothing here for "soft" targets if needed,
                # but hard 1.0 works for BCE Loss.
                target_tensor[0, rec_days] = 1.0

            self.samples.append((input_tensor, target_tensor))

        logger.info(f"Created {len(self.samples)} samples from {len(grouped)} accounts.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# --- 2. The Model (1D U-Net) ---

class SignalCleaner(nn.Module):
    """
    A 1D U-Net that learns to 'denoise' transaction timelines.
    It captures local patterns (Conv) and global periodicity (Bottleneck).
    """

    def __init__(self):
        super().__init__()

        # Encoder: Compresses time (180 -> 90 -> 45)
        # Detects patterns like "activity every 7 steps"
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Bottleneck: High-level context (e.g., "User is weekly")
        self.center = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # Decoder: Reconstructs the 'Clean' timeline (45 -> 90 -> 180)
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
            # Input channels = 32 (from dec2) + 16 (skip connection from enc1)
            nn.Conv1d(32 + 16, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )

        # Final Classifier per timestep
        # Input: 16 (from dec1) + 1 (original input skip) -> Output: 1 probability
        self.final = nn.Conv1d(16 + 1, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # [B, 16, L/2]
        e2 = self.enc2(e1)  # [B, 32, L/4]

        # Bottleneck
        c = self.center(e2)  # [B, 64, L/4]

        # Decoder with Skip Connections
        d2 = self.dec2(c)  # [B, 32, L/2]

        # Skip connection: Add e1 (low-level features) to refine timing
        # We concatenate along channel dim (dim 1)
        # Note: Upsampling might cause off-by-one length mismatch due to odd/even split.
        # We simply slice to match size if needed.
        if d2.size(2) != e1.size(2):
            d2 = torch.nn.functional.interpolate(d2, size=e1.size(2))

        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # [B, 16, L]

        # Final Skip: Add original input so model knows EXACTLY where events happened
        if d1.size(2) != x.size(2):
            d1 = torch.nn.functional.interpolate(d1, size=x.size(2))

        out = self.final(torch.cat([d1, x], dim=1))  # [B, 1, L]

        return torch.sigmoid(out)


# --- 3. Training Logic ---

def train_learner(
        epochs=20,
        batch_size=32,
        lr=1e-3,
        seq_len=180,
        data_path='data/rec_data2.csv',
        save_path='cache/neural_ts_model.pt'
):
    # 1. Load Data
    field_config = FieldConfig()
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=[field_config.date, field_config.label])

    # 2. Build Dataset & Loader
    dataset = TimeSeriesDataset(df, field_config, seq_len=seq_len)

    # Split Train/Val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3. Initialize Model
    model = SignalCleaner().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Weighted BCE Loss?
    # Recurring days are rare (sparse). We might want to weight positive samples higher.
    # For now, standard BCELoss is a good baseline.
    criterion = nn.BCELoss()

    logger.info(f"Starting training on {DEVICE} for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)
                val_preds = model(x_val)
                v_loss = criterion(val_preds, y_val)
                val_loss += v_loss.item()

        avg_val_loss = val_loss / len(val_loader)

        logger.info(f"Epoch {epoch + 1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 4. Save Model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    train_learner()