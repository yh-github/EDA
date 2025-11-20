import logging
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import List, Literal, Tuple

from config import FieldConfig

logger = logging.getLogger(__name__)


@dataclass
class TSConfig:
    # Signal Prep
    signal_mode: Literal['amount', 'count', 'binary'] = 'binary'
    min_history_days: int = 14

    # Spectral Filtering
    # "How many distinct cycles to keep?" (e.g. Monthly + BiWeekly + Weekly = 3)
    max_harmonics: int = 3
    # "Ignore frequencies weaker than X% of the strongest one"
    spectral_threshold: float = 0.2

    # Detection
    # "How high must the reconstructed wave be to trigger a date?"
    detection_threshold: float = 0.4
    # "How much jitter (days) is allowed for a match?"
    jitter_tolerance: int = 1


class RobustTSAnalyzer:
    """
    Uses FFT Spectral Filtering to separate 'Signal' (Recurring) from 'Noise' (Random).
    Reconstructs a clean timeline to predict specific dates.
    """

    def __init__(self, field_config: FieldConfig, config: TSConfig = TSConfig()):
        self.fc = field_config
        self.cfg = config

    def _get_daily_signal(self, df: pd.DataFrame) -> Tuple[pd.DatetimeIndex, np.ndarray]:
        """Converts dataframe to dense daily signal."""
        if df.empty:
            return pd.DatetimeIndex([]), np.array([])

        dates = pd.to_datetime(df[self.fc.date])
        start_date = dates.min().normalize()
        end_date = dates.max().normalize()

        all_days = pd.date_range(start_date, end_date, freq='D')
        if len(all_days) < self.cfg.min_history_days:
            return pd.DatetimeIndex([]), np.array([])

        temp_df = df.copy()
        temp_df['date_norm'] = dates.dt.normalize()

        if self.cfg.signal_mode == 'amount':
            daily = temp_df.groupby('date_norm')[self.fc.amount].sum()
            # Log compression to handle massive outliers
            daily = np.log1p(np.abs(daily))
        elif self.cfg.signal_mode == 'binary':
            # 1.0 if ANY recurrence happened, 0.0 otherwise
            daily = temp_df.groupby('date_norm').size().clip(upper=1)
        else:
            daily = temp_df.groupby('date_norm').size()

        signal = daily.reindex(all_days, fill_value=0.0).values
        return all_days, signal

    def predict_dates(self, df: pd.DataFrame) -> List[pd.Timestamp]:
        """
        Returns a list of dates where a recurring event is predicted.
        """
        time_index, signal = self._get_daily_signal(df)
        if len(signal) == 0: return []

        # 1. FFT (Time -> Frequency)
        # rfft is for real-valued inputs (returns complex hermitian)
        fft_coeffs = np.fft.rfft(signal)

        # 2. Spectral Filtering (Keep only strong periodic components)
        magnitudes = np.abs(fft_coeffs)
        # Zero out DC component (overall average) for peak finding
        magnitudes[0] = 0

        # Find top K strongest frequencies
        # indices of sorted magnitudes (descending)
        top_indices = np.argsort(magnitudes)[::-1]

        # Create a clean filter mask
        mask = np.zeros_like(fft_coeffs, dtype=bool)

        # Always keep DC (index 0) for reconstruction baseline,
        # but we don't count it as a "harmonic"
        mask[0] = True

        max_power = magnitudes[top_indices[0]] if len(top_indices) > 0 else 0

        found_harmonics = 0
        for idx in top_indices:
            if idx == 0: continue

            # If this frequency is strong enough
            if magnitudes[idx] >= max_power * self.cfg.spectral_threshold:
                mask[idx] = True
                found_harmonics += 1

            if found_harmonics >= self.cfg.max_harmonics:
                break

        # Apply mask
        clean_fft = fft_coeffs * mask

        # 3. Inverse FFT (Frequency -> Clean Time Signal)
        # This creates a "perfect" wave made of only the dominant cycles
        clean_signal = np.fft.irfft(clean_fft, n=len(signal))

        # 4. Peak Detection (Clean Signal -> Dates)
        # We use the configured threshold relative to the signal's max
        if clean_signal.max() > 0:
            # Normalize 0-1 for consistent thresholding
            clean_signal_norm = clean_signal / clean_signal.max()
            peaks, _ = find_peaks(clean_signal_norm, height=self.cfg.detection_threshold)
        else:
            peaks = []

        predicted_dates = time_index[peaks].tolist()
        return predicted_dates
