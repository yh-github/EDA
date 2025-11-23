import logging
import random
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader

# Reuse your config
from common.config import FieldConfig
from glob.hyper_tuner import HyperTuner  # For data loading logic

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_training_pairs(df: pd.DataFrame, field_config: FieldConfig):
    """
    Generates Positive (Similar) and Negative (Dissimilar) pairs
    based on Amount and isRecurring heuristics.
    """
    examples = []

    # Group by Account to ensure we only compare a user's own transactions
    grouped = df.groupby(field_config.accountId)

    pos_count = 0
    neg_count = 0

    for acc_id, group in grouped:
        # Filter to only Recurring transactions
        # (We only care about clustering the recurring ones correctly)
        rec_txns = group[group[field_config.label] == True]

        if len(rec_txns) < 2:
            continue

        texts = rec_txns[field_config.text].tolist()
        amounts = rec_txns[field_config.amount].values

        # Create pairs (Vectorized or simple loop)
        # Simple loop for clarity:
        n = len(rec_txns)

        # Heuristic Constants
        # If amounts are within $0.50, we assume same subscription (Positive)
        # If amounts differ by > $2.00, we assume different subscription (Negative)
        SAME_AMT_THRESH = 0.50
        DIFF_AMT_THRESH = 2.00

        # Limit pairs per account to avoid explosion
        max_pairs = 50
        current_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                if current_pairs >= max_pairs: break

                amt_diff = abs(amounts[i] - amounts[j])
                text_a = texts[i]
                text_b = texts[j]

                # POSITIVE PAIR
                if amt_diff <= SAME_AMT_THRESH:
                    # label=1.0 means "These should be close"
                    examples.append(InputExample(texts=[text_a, text_b], label=1.0))
                    pos_count += 1
                    current_pairs += 1

                # NEGATIVE PAIR
                elif amt_diff > DIFF_AMT_THRESH:
                    # label=0.0 means "These should be far apart"
                    examples.append(InputExample(texts=[text_a, text_b], label=0.0))
                    neg_count += 1
                    current_pairs += 1

    logger.info(f"Generated {len(examples)} pairs. Pos: {pos_count}, Neg: {neg_count}")
    return examples


def main():
    # 1. Load Data
    field_config = FieldConfig()
    # Load using your existing util
    df_train_val, _ = HyperTuner.load_and_split_data(
        filter_direction=1,
        data_path=Path('data/rec_data2.csv'),
        field_config=field_config,
        random_state=42
    )

    # 2. Generate Pairs
    train_examples = generate_training_pairs(df_train_val, field_config)

    # Shuffle and Split (Train/Val)
    random.shuffle(train_examples)
    split_idx = int(len(train_examples) * 0.9)
    train_data = train_examples[:split_idx]
    val_data = train_examples[split_idx:]

    # 3. Initialize Model
    # We start with the generic MPNET
    model_name = "sentence-transformers/all-mpnet-base-v2"
    logger.info(f"Loading base model: {model_name}")
    model = SentenceTransformer(model_name)

    # 4. Define Loss
    # ContrastiveLoss pushes positives close and negatives apart
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=32)
    train_loss = losses.ContrastiveLoss(model=model)

    # 5. Evaluator (Checks correlation on validation set)
    # BinaryClassificationEvaluator checks if cosine sim predicts the label
    evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples(
        val_data, name='val_pairs'
    )

    # 6. Train
    logger.info("Starting Fine-Tuning...")
    output_path = "ft_models/fine_tuned_mpnet"

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=1,  # 1 epoch is often enough for this
        warmup_steps=100,
        output_path=output_path,
        show_progress_bar=True
    )

    logger.info(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()