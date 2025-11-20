import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from scipy.signal import find_peaks

from config import FieldConfig, get_device
from log_utils import setup_logging
from ts_neural_cleaner import SignalCleaner  # Import your model class

setup_logging(Path('logs/'), "eval_neural")
logger = logging.getLogger(__name__)
DEVICE = get_device()


class NeuralAnalyzerAdapter:
    """
    Wraps the PyTorch SignalCleaner to look like the RobustTSAnalyzer.
    (Takes a DataFrame -> Returns a list of Dates)
    """

    @dataclass
    class Params:
        jitter_tolerance: int = 1  # Allow +/- 1 day match

    def __init__(self, model_path, seq_len=180):
        self.field_config = FieldConfig()
        self.seq_len = seq_len
        self.model = SignalCleaner().to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        self.cfg = NeuralAnalyzerAdapter.Params()


    def predict_dates(self, df: pd.DataFrame):
        # 1. Prep Input (Same logic as training dataset)
        dates = pd.to_datetime(df[self.field_config.date])
        start_date = dates.min().normalize()
        days = (dates - start_date).dt.days.values

        # Create sparse signal
        input_tensor = torch.zeros((1, 1, self.seq_len), dtype=torch.float32).to(DEVICE)
        valid_idx = days[days < self.seq_len]
        if len(valid_idx) == 0: return []

        input_tensor[0, 0, valid_idx] = 1.0

        # 2. Inference
        with torch.no_grad():
            # Output is [1, 1, Seq_Len] probability map (0.0 to 1.0)
            probs = self.model(input_tensor).cpu().numpy()[0, 0]

        # 3. Peak Detection (Probabilities -> Dates)
        # Threshold: 0.2 (adjustable)
        peaks, _ = find_peaks(probs, height=0.2, distance=5)

        # Convert day indices back to Real Dates
        predicted_dates = [start_date + pd.Timedelta(days=int(p)) for p in peaks]
        return predicted_dates


# --- Reuse the exact evaluation logic from ts_eval_dates.py ---
# (You can import evaluate_account if you refactor, or copy-paste it here)
from ts_eval_dates import evaluate_account


def main():
    fc = FieldConfig()
    df = pd.read_csv("data/rec_data2.csv")
    df = df.dropna(subset=[fc.date, fc.amount, fc.text])

    # 1. Load the Wrapper
    # Ensure you ran 'python ts_neural_cleaner.py' first to generate this file!
    analyzer = NeuralAnalyzerAdapter(model_path="cache/neural_ts_model.pt")

    total_tp, total_fp, total_fn = 0, 0, 0
    accounts = df[fc.accountId].unique()

    logger.info(f"Evaluating Neural Model on {len(accounts)} accounts...")

    for acc_id in accounts:
        acc_df = df[df[fc.accountId] == acc_id]
        if len(acc_df) < 14: continue  # Skip too short

        # This now calls the Neural Network to find dates
        tp, fp, fn = evaluate_account(acc_df, fc, analyzer)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    logger.info(f"NEURAL RESULTS | F1: {f1:.4f} | P: {precision:.4f} | R: {recall:.4f}")


if __name__ == "__main__":
    main()