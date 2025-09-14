import pandas as pd
import numpy as np
from scipy import signal
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

class EEGPreprocessor:
    def __init__(self, sample_rate: int = 250):
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2

        self.channel_pairs = {
            'C3-C4': ('C3', 'C4'),
            'P3-P4': ('P3', 'P4'),
            'P7-P8': ('P7', 'P8'),
            'T7-T8': ('T7', 'T8')
        }

        self.frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }

        self.filters = self._create_filters()

    def _create_filters(self) -> Dict[str, Tuple]:
        filters = {}
        for band_name, (low, high) in self.frequency_bands.items():
            if high > self.nyquist:
                high = self.nyquist - 1
            sos = signal.butter(4, [low, high], btype='band', fs=self.sample_rate, output='sos')
            filters[band_name] = sos
        return filters

    def detect_sample_rate(self, filepath: str) -> int:
        """Detect sample rate from CSV header"""
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i >= 4:  # Only check first 4 lines
                    break
                if 'Sample Rate' in line:
                    # Extract number from line like "%Sample Rate = 500 Hz"
                    import re
                    match = re.search(r'(\d+)\s*Hz', line)
                    if match:
                        return int(match.group(1))
        return self.sample_rate  # Return default if not found

    def load_data(self, filepath: str) -> pd.DataFrame:
        # Auto-detect and update sample rate if different
        detected_rate = self.detect_sample_rate(filepath)
        if detected_rate != self.sample_rate:
            print(f"Detected sample rate {detected_rate}Hz, updating from {self.sample_rate}Hz")
            self.sample_rate = detected_rate
            self.nyquist = detected_rate / 2
            self.filters = self._create_filters()  # Recreate filters for new rate

        df = pd.read_csv(filepath, skiprows=4)

        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace('EOG ', '', regex=False)

        # Check if we have 8-channel or 16-channel data
        num_channels = len([col for col in df.columns if col.startswith('Channel')])

        if num_channels == 8:
            # New Setup v2: 8 channels at 500Hz
            # Only drop non-EEG columns
            columns_to_drop = ['Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2', 'Not Used',
                             'Digital Channel 0 (D11)', 'Digital Channel 1 (D12)', 'Digital Channel 2 (D13)',
                             'Digital Channel 3 (D17)', 'Not Used.1', 'Digital Channel 4 (D18)',
                             'Analog Channel 0', 'Analog Channel 1', 'Analog Channel 2']
            df = df.drop(columns=columns_to_drop, errors='ignore')

            # New channel mapping for 8-channel setup
            channel_mapping = {
                'Channel 0': 'T7',
                'Channel 1': 'T8',
                'Channel 2': 'C3',
                'Channel 3': 'C4',
                'Channel 4': 'P7',
                'Channel 5': 'P8',
                'Channel 6': 'P3',
                'Channel 7': 'P4'
            }
        else:
            # Original Setup v1: 16 channels at 250Hz
            columns_to_drop = ['Channel 0', 'Channel 1'] + [f'Channel {i}' for i in range(10, 16)] + \
                             ['Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2', 'Not Used',
                              'Digital Channel 0 (D11)', 'Digital Channel 1 (D12)', 'Digital Channel 2 (D13)',
                              'Digital Channel 3 (D17)', 'Not Used.1', 'Digital Channel 4 (D18)',
                              'Analog Channel 0', 'Analog Channel 1', 'Analog Channel 2']
            df = df.drop(columns=columns_to_drop, errors='ignore')

            # Original channel mapping for 16-channel setup
            channel_mapping = {
                'Channel 2': 'C3',
                'Channel 3': 'C4',
                'Channel 4': 'P7',
                'Channel 5': 'P8',
                'Channel 6': 'P3',
                'Channel 7': 'P4',
                'Channel 8': 'T7',
                'Channel 9': 'T8'
            }

        df = df.rename(columns=channel_mapping)

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        if 'Marker' in df.columns:
            marker_indices = df[df['Marker'] != 0].index

            # Adjust marker propagation windows based on sampling rate
            if self.sample_rate == 250:
                # For 250Hz: standard window sizes
                before_window = 49  # ~0.2 seconds
                after_window = 200  # ~0.8 seconds
            else:
                # For other rates: scale proportionally
                before_window = int(49 * self.sample_rate / 250)
                after_window = int(200 * self.sample_rate / 250)

            for index in marker_indices:
                marker_value = df.loc[index, 'Marker']
                start_index = max(0, index - before_window)
                end_index = min(len(df) - 1, index + after_window)

                mask = (df.loc[start_index:end_index, 'Marker'] == 0)
                df.loc[start_index:end_index, 'Marker'] = df.loc[start_index:end_index, 'Marker'].where(~mask, marker_value)

        return df

    def normalize_channels(self, df: pd.DataFrame, channels: List[str]) -> pd.DataFrame:
        df_normalized = df.copy()
        for channel in channels:
            if channel in df.columns:
                mean_val = df[channel].mean()
                std_val = df[channel].std()
                if std_val > 0:
                    df_normalized[channel] = (df[channel] - mean_val) / std_val
        return df_normalized

    def apply_bandpass_filter(self, signal_data: np.ndarray, band: str) -> np.ndarray:
        if band not in self.filters:
            raise ValueError(f"Unknown frequency band: {band}")

        if len(signal_data) < 13:
            return signal_data

        filtered = signal.sosfiltfilt(self.filters[band], signal_data)
        return filtered

    def compute_channel_differences(self, df: pd.DataFrame) -> pd.DataFrame:
        diff_df = pd.DataFrame()

        for pair_name, (ch1, ch2) in self.channel_pairs.items():
            if ch1 in df.columns and ch2 in df.columns:
                diff_df[pair_name] = df[ch1] - df[ch2]

        if 'Marker' in df.columns:
            diff_df['Marker'] = df['Marker']
        if 'Timestamp' in df.columns:
            diff_df['Timestamp'] = df['Timestamp']

        return diff_df

    def filter_all_bands(self, signal_data: np.ndarray) -> Dict[str, np.ndarray]:
        filtered_bands = {}
        for band in self.frequency_bands.keys():
            filtered_bands[band] = self.apply_bandpass_filter(signal_data, band)
        return filtered_bands

    def preprocess_pipeline(self, filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("Loading data...")
        df = self.load_data(filepath)

        eeg_channels = ['C3', 'C4', 'P3', 'P4', 'P7', 'P8', 'T7', 'T8']
        existing_channels = [ch for ch in eeg_channels if ch in df.columns]

        print(f"Normalizing {len(existing_channels)} EEG channels...")
        df_normalized = self.normalize_channels(df, existing_channels)

        print("Computing channel differences...")
        df_differences = self.compute_channel_differences(df_normalized)

        return df_normalized, df_differences

def main():
    # Default to 250Hz for 8-channel setup
    preprocessor = EEGPreprocessor(sample_rate=250)

    df_normalized, df_differences = preprocessor.preprocess_pipeline(
        "OpenBCI-RAW-2025-09-13_23-58-04.csv"
    )

    print(f"\nNormalized data shape: {df_normalized.shape}")
    print(f"Channel differences shape: {df_differences.shape}")
    print(f"\nChannel difference columns: {df_differences.columns.tolist()}")

    non_zero_markers = (df_differences['Marker'] > 0).sum() if 'Marker' in df_differences.columns else 0
    print(f"Samples with touch markers: {non_zero_markers}")

if __name__ == "__main__":
    main()