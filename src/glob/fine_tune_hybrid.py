import logging
import random
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from classifier import HybridModel
# --- Project Imports ---
from common.config import FieldConfig, EmbModel, get_device
from common.embedder import EmbeddingService
from common.feature_processor import HybridFeatureProcessor, FeatProcParams, FeatureHyperParams
from common.log_utils import setup_logging

# --- Setup ---
setup_logging(Path('logs/'), "fine_tune_hybrid")
logger = logging.getLogger(__name__)
DEVICE = get_device()


# --- 1. Custom Dataset ---

class HybridPairDataset(Dataset):
    """
    Yields pairs of transaction indices (idx_a, idx_b) and a label.
    label = 1.0 (Similar), 0.0 (Dissimilar)
    """

    def __init__(self,
                 pair_indices: List[Tuple[int, int, float]],
                 x_text: torch.Tensor,
                 x_continuous: torch.Tensor,
                 x_categorical: torch.Tensor):
        self.pairs = pair_indices
        self.x_text = x_text
        self.x_continuous = x_continuous
        self.x_categorical = x_categorical

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx_a, idx_b, label = self.pairs[idx]

        # Helper to extract one sample's features
        def get_features(i):
            return (
                self.x_text[i],
                self.x_continuous[i],
                self.x_categorical[i]
            )

        feat_a = get_features(idx_a)
        feat_b = get_features(idx_b)

        return feat_a, feat_b, torch.tensor(label, dtype=torch.float32)


# --- 2. Pair Generation Logic ---

def generate_training_pairs(df: pd.DataFrame, field_config: FieldConfig) -> List[Tuple[int, int, float]]:
    """
    Generates indices of Positive and Negative pairs using heuristic logic.
    Positive: Same Account, Recurring=True, Similar Amount (<= $0.50 diff)
    Negative: Same Account, Recurring=True, Diff Amount (> $2.00 diff)
    """
    logger.info("Generating training pairs...")
    pairs = []

    df = df.reset_index(drop=True)
    recurring_mask = df[field_config.label] == 1
    recurring_indices = df.index[recurring_mask].tolist()
    acc_map = df.loc[recurring_indices].groupby(field_config.accountId).groups

    pos_count = 0
    neg_count = 0

    SAME_AMT_THRESH = 0.50
    DIFF_AMT_THRESH = 2.00
    MAX_PAIRS_PER_ACC = 100

    for acc_id, indices in acc_map.items():
        indices = list(indices)
        n = len(indices)
        if n < 2: continue

        random.shuffle(indices)
        amounts = df.loc[indices, field_config.amount].values

        acc_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                if acc_pairs >= MAX_PAIRS_PER_ACC: break

                idx_a = indices[i]
                idx_b = indices[j]

                amt_a = amounts[i]
                amt_b = amounts[j]
                diff = abs(amt_a - amt_b)

                if diff <= SAME_AMT_THRESH:
                    pairs.append((idx_a, idx_b, 1.0))
                    pos_count += 1
                    acc_pairs += 1
                elif diff > DIFF_AMT_THRESH:
                    pairs.append((idx_a, idx_b, 0.0))
                    neg_count += 1
                    acc_pairs += 1

    logger.info(f"Generated {len(pairs)} pairs. Positive: {pos_count}, Negative: {neg_count}")
    return pairs


# --- 3. Siamese Network & Loss ---

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
                                                              2))
        return loss_contrastive


# --- 4. Training Loop ---

def train(model, dataloader, num_epochs=5, learning_rate=1e-4, save_path=None):
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(DEVICE)

    logger.info(f"Starting training on {DEVICE}...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (feat_a, feat_b, labels) in enumerate(dataloader):
            x_text_a, x_cont_a, x_cat_a = [t.to(DEVICE) for t in feat_a]
            x_text_b, x_cont_b, x_cat_b = [t.to(DEVICE) for t in feat_b]
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            emb_a = model.embed(x_text_a, x_cont_a, x_cat_a)
            emb_b = model.embed(x_text_b, x_cont_b, x_cat_b)

            loss = criterion(emb_a, emb_b, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model state saved to {save_path}")


# --- 5. Main Execution ---

def main():
    # Configuration
    DATA_PATH = Path('data/rec_data2.csv')
    OUTPUT_DIR = Path('ft_models/mlp_hybrid')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH = OUTPUT_DIR / 'mlp_model_state.pt'

    field_config = FieldConfig()

    # 1. Load Data
    df_full = pd.read_csv(DATA_PATH)
    df_full = df_full.dropna(subset=[field_config.date, field_config.amount, field_config.text])
    logger.info(f"Loaded {len(df_full)} rows.")

    # 2. Feature Engineering
    feat_params = FeatProcParams(
        use_cyclical_dates=True,
        use_continuous_amount=True,
        use_categorical_amount=False
    )

    processor = HybridFeatureProcessor.create(feat_params, field_config)
    metadata = processor.fit(df_full)
    processed_df = processor.transform(df_full)

    # 3. Base Embeddings (Text)
    emb_service = EmbeddingService(model_name=EmbModel.MPNET, max_length=64, batch_size=256)
    all_texts = df_full[field_config.text].tolist()
    x_text_np = emb_service.embed(all_texts)

    # 4. Prepare Tensor Data
    x_text = torch.from_numpy(x_text_np).float()

    # Continuous (Normalize!)
    scaler = None  # Initialize variable
    if metadata.continuous_scalable_cols:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        # Extract Cyclical
        x_cyc = processed_df[metadata.cyclical_cols].values if metadata.cyclical_cols else np.zeros((len(df_full), 0))

        # Extract and Scale Continuous
        x_cont_raw = processed_df[
            metadata.continuous_scalable_cols].values if metadata.continuous_scalable_cols else np.zeros(
            (len(df_full), 0))
        if x_cont_raw.shape[1] > 0:
            x_cont_raw = scaler.fit_transform(x_cont_raw)

        x_continuous_np = np.concatenate([x_cyc, x_cont_raw], axis=1)
    else:
        x_continuous_np = np.zeros((len(df_full), 0))

    x_continuous = torch.from_numpy(x_continuous_np).float()

    # Categorical
    cat_cols = list(metadata.categorical_features.keys())
    if cat_cols:
        x_cat_np = processed_df[cat_cols].values
    else:
        x_cat_np = np.zeros((len(df_full), 0))

    x_categorical = torch.from_numpy(x_cat_np).long()

    # 5. Build Model
    from common.data import FeatureSet
    dummy_fs = FeatureSet(x_text_np, x_continuous_np, x_cat_np, np.zeros(len(df_full)))
    feature_config = FeatureHyperParams.build(dummy_fs, metadata)

    mlp_config = HybridModel.MlpHyperParams(
        text_projection_dim=128,
        mlp_hidden_layers=[128, 64],
        dropout_rate=0.2
    )

    model = HybridModel(feature_config, mlp_config)

    # 6. Generate Pairs
    pair_indices = generate_training_pairs(df_full, field_config)

    if not pair_indices:
        logger.error("No pairs generated. Check data or heuristics.")
        return

    # 7. Create Dataset & Loader
    dataset = HybridPairDataset(pair_indices, x_text, x_continuous, x_categorical)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 8. Train
    train(model, dataloader, num_epochs=5, learning_rate=1e-4, save_path=MODEL_PATH)

    # 9. Save Processor and Scaler for Inference
    logger.info("Saving inference artifacts...")

    # Save Processor (contains vocab maps, bin edges, etc.)
    processor_path = OUTPUT_DIR / 'processor.joblib'
    joblib.dump(processor, processor_path)
    logger.info(f"Processor saved to {processor_path}")

    # Save Scaler (contains mean/std for continuous features)
    if scaler is not None:
        scaler_path = OUTPUT_DIR / 'scaler.joblib'
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
    else:
        logger.info("No scaler was used/created.")


if __name__ == "__main__":
    main()