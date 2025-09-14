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

    def load_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath, skiprows=4)

        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace('EOG ', '', regex=False)

        columns_to_drop = ['Channel 0', 'Channel 1'] + [f'Channel {i}' for i in range(10, 16)] + \
                         ['Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2', 'Not Used',
                          'Digital Channel 0 (D11)', 'Digital Channel 1 (D12)', 'Digital Channel 2 (D13)',
                          'Digital Channel 3 (D17)', 'Not Used.1', 'Digital Channel 4 (D18)',
                          'Analog Channel 0', 'Analog Channel 1', 'Analog Channel 2']
        df = df.drop(columns=columns_to_drop, errors='ignore')

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
            before_window = 49
            after_window = 200

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