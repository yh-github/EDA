import pandas as pd
import torch
from dataclasses import dataclass
from pytorch_forecasting import TemporalFusionTransformer


# We map the raw string "global_group_id" back to business logic
@dataclass
class InteroperableGroup:
    """
    The final business object describing a recurring pattern.
    Compatible with the 'RecurringGroupResult' concept.
    """
    group_id: str  # e.g. "acc123_amt_15.99"
    description: str  # e.g. "NETFLIX.COM"
    amount_stats: float  # Median amount (robust)
    cycle_days: float  # Median days between transactions
    confidence: float  # Aggregated model probability (0.0 - 1.0)
    is_recurring: bool  # Final hard label based on threshold


class TFTGroupExtractor:
    """
    Wraps a trained TFT model to convert sequence predictions into
    structured, interoperable group objects.
    """

    def __init__(self, model: TemporalFusionTransformer, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold

        # Retrieve the encoder to decode integer IDs back to strings like "acc1_amt_15.99"
        # In your config, the group identifier is 'global_group_id'
        self.group_encoder = model.dataset.categorical_encoders['global_group_id']

    def extract(
            self,
            raw_predictions: torch.Tensor,
            x: dict[str, torch.Tensor],
            original_df: pd.DataFrame
    ) -> list[InteroperableGroup]:
        """
        Args:
            raw_predictions: Output from model.predict(..., mode="raw")['prediction']
                             Shape: [Batch, Prediction_Length, Classes]
            x: The input dictionary returned by predict(..., return_x=True).
               Contains encoded categorical IDs.
            original_df: The source DataFrame (from tft_data_clustered.py) containing
                         actual text descriptions and dates.
        """
        results: list[InteroperableGroup] = []

        # 1. Extract Group IDs from the input tensors
        # In TFT, static categoricals are usually at specific indices.
        # We assume 'global_group_id' is the first static categorical based on your config.
        # Shape of x['decoder_cat'] is [Batch, Time, n_categoricals]
        # We only need the first time step's ID for the group.
        encoded_group_ids = x['decoder_cat'][:, 0, 0].cpu().numpy()
        decoded_group_ids = self.group_encoder.inverse_transform(encoded_group_ids)

        # 2. Extract Probabilities for the "Recurring" class (Index 1)
        # Assuming binary classification output_size=[2]
        # We take the mean probability across the prediction horizon (if > 1 step)
        recurrence_probs = torch.softmax(raw_predictions, dim=-1)[:, :, 1]
        mean_probs = recurrence_probs.mean(dim=1).cpu().numpy()

        # 3. Optimization: Pre-group the original dataframe for fast lookups
        # This avoids boolean indexing inside the loop (O(N) -> O(1))
        df_grouped = original_df.groupby('global_group_id')

        # 4. Iterate and Construct Objects
        for group_id, confidence in zip(decoded_group_ids, mean_probs):

            # Skip low confidence groups immediately if you want speed
            # if confidence < 0.1: continue

            if group_id not in df_grouped.groups:
                continue

            # Get raw data for this cluster
            group_rows = df_grouped.get_group(group_id)

            # --- Interoperability 1: Text Description ---
            # Use Mode (most frequent) to ignore noise like "Netflix *123" vs "Netflix *124"
            text_mode = group_rows['bankRawDescription'].mode()
            description = text_mode[0] if not text_mode.empty else "Unknown"

            # --- Interoperability 2: Amount Stats ---
            # We use Median as it's robust to outliers (e.g., one-time fees)
            median_amount = float(group_rows['amount'].median())

            # --- Interoperability 3: Cycle (Frequency) ---
            # Calculate diff between sorted dates
            dates = group_rows['date'].sort_values()
            if len(dates) > 1:
                diffs = dates.diff().dt.days.dropna()
                cycle_days = float(diffs.median())
            else:
                cycle_days = 0.0

            # Create the result object
            result = InteroperableGroup(
                group_id=str(group_id),
                description=str(description),
                amount_stats=median_amount,
                cycle_days=cycle_days,
                confidence=float(confidence),
                is_recurring=bool(confidence >= self.threshold)
            )

            results.append(result)

        return results