import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import signal as scipy_signal
from scipy import stats
from preprocess import EEGPreprocessor

class FeatureExtractor:
    def __init__(self, sample_rate: int = 250, window_size: int = 250):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.preprocessor = EEGPreprocessor(sample_rate)

    def extract_window_features(self, window_data: np.ndarray, band_name: str) -> Dict[str, float]:
        features = {}

        filtered_signal = self.preprocessor.apply_bandpass_filter(window_data, band_name)

        power = np.square(filtered_signal)
        features[f'{band_name}_mean_power'] = np.mean(power)
        features[f'{band_name}_std'] = np.std(filtered_signal)
        features[f'{band_name}_peak_to_peak'] = np.ptp(filtered_signal)

        psd_freqs, psd_power = scipy_signal.periodogram(filtered_signal, fs=self.sample_rate)
        band_low, band_high = self.preprocessor.frequency_bands[band_name]
        band_mask = (psd_freqs >= band_low) & (psd_freqs <= band_high)
        features[f'{band_name}_band_power'] = np.sum(psd_power[band_mask])

        features[f'{band_name}_skewness'] = stats.skew(filtered_signal)
        features[f'{band_name}_kurtosis'] = stats.kurtosis(filtered_signal)

        return features

    def compute_band_power_ratios(self, all_band_features: Dict[str, float]) -> Dict[str, float]:
        ratios = {}

        bands = list(self.preprocessor.frequency_bands.keys())
        for i, band1 in enumerate(bands):
            for band2 in bands[i+1:]:
                power1 = all_band_features.get(f'{band1}_band_power', 0.001)
                power2 = all_band_features.get(f'{band2}_band_power', 0.001)
                ratios[f'{band1}_{band2}_ratio'] = power1 / power2

        return ratios

    def extract_all_features_for_window(self, window_data: np.ndarray) -> Dict[str, float]:
        all_features = {}

        for band_name in self.preprocessor.frequency_bands.keys():
            band_features = self.extract_window_features(window_data, band_name)
            all_features.update(band_features)

        ratios = self.compute_band_power_ratios(all_features)
        all_features.update(ratios)

        return all_features

    def create_sliding_windows(self, data: np.ndarray, labels: Optional[np.ndarray] = None,
                             overlap: float = 0.5) -> Tuple[List[np.ndarray], Optional[List[int]]]:
        step_size = int(self.window_size * (1 - overlap))
        windows = []
        window_labels = [] if labels is not None else None

        for start in range(0, len(data) - self.window_size + 1, step_size):
            end = start + self.window_size
            windows.append(data[start:end])

            if labels is not None:
                window_marker = labels[start:end]
                label = 1 if np.any(window_marker > 0) else 0
                window_labels.append(label)

        return windows, window_labels

    def extract_features_from_dataframe(self, df: pd.DataFrame,
                                       channel_pairs: Optional[List[str]] = None,
                                       overlap: float = 0.5) -> Tuple[pd.DataFrame, np.ndarray]:
        if channel_pairs is None:
            channel_pairs = ['C3-C4', 'P3-P4', 'P7-P8', 'T7-T8']

        all_features = []
        all_labels = []

        markers = df['Marker'].values if 'Marker' in df.columns else None

        for pair in channel_pairs:
            if pair not in df.columns:
                print(f"Warning: Channel pair {pair} not found in dataframe")
                continue

            signal_data = df[pair].values
            windows, window_labels = self.create_sliding_windows(signal_data, markers, overlap)

            print(f"Processing {len(windows)} windows for {pair}...")
            for i, window in enumerate(windows):
                window_features = self.extract_all_features_for_window(window)

                for key in list(window_features.keys()):
                    window_features[f'{pair}_{key}'] = window_features.pop(key)

                all_features.append(window_features)

            if window_labels:
                all_labels.extend(window_labels)

        features_df = pd.DataFrame(all_features)
        labels_array = np.array(all_labels) if all_labels else None

        return features_df, labels_array

    def extract_features_pipeline(self, preprocessed_df: pd.DataFrame,
                                 overlap: float = 0.5) -> Tuple[pd.DataFrame, np.ndarray]:
        print("Extracting features from all channel pairs...")
        features_df, labels = self.extract_features_from_dataframe(
            preprocessed_df,
            overlap=overlap
        )

        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)

        print(f"Feature extraction complete:")
        print(f"  Total features: {features_df.shape[1]}")
        print(f"  Total samples: {features_df.shape[0]}")
        if labels is not None:
            print(f"  Touch samples: {np.sum(labels == 1)}")
            print(f"  No-touch samples: {np.sum(labels == 0)}")

        return features_df, labels

def main():
    from preprocess import EEGPreprocessor

    preprocessor = EEGPreprocessor(sample_rate=250)
    _, df_differences = preprocessor.preprocess_pipeline(
        "OpenBCI-RAW-2025-09-13_23-58-04.csv"
    )

    extractor = FeatureExtractor(sample_rate=250, window_size=250)
    features_df, labels = extractor.extract_features_pipeline(df_differences, overlap=0.5)

    print(f"\nFeatures shape: {features_df.shape}")
    print(f"Labels shape: {labels.shape if labels is not None else 'None'}")

    print("\nSample features (first 5 columns):")
    print(features_df.iloc[:5, :5])

    if labels is not None:
        print(f"\nClass distribution:")
        print(f"  No-touch (0): {np.sum(labels == 0)} ({100*np.mean(labels == 0):.1f}%)")
        print(f"  Touch (1): {np.sum(labels == 1)} ({100*np.mean(labels == 1):.1f}%)")

if __name__ == "__main__":
    main()