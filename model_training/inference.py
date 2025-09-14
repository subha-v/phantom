import numpy as np
import joblib
from collections import deque
from typing import Dict, Tuple, Optional, List
import time
from preprocess import EEGPreprocessor
from features import FeatureExtractor

class RealTimeInference:
    def __init__(self, model_path: str = 'touch_detection_model.pkl',
                 sample_rate: int = 250, window_size: int = 250,
                 prediction_interval: int = 10, voting_window: int = 5):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.prediction_interval = prediction_interval
        self.voting_window = voting_window

        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_name = model_data.get('model_name', 'Unknown')

        self.preprocessor = EEGPreprocessor(sample_rate)
        self.feature_extractor = FeatureExtractor(sample_rate, window_size)

        self.buffers = {
            'C3-C4': deque(maxlen=window_size),
            'P3-P4': deque(maxlen=window_size),
            'P7-P8': deque(maxlen=window_size),
            'T7-T8': deque(maxlen=window_size)
        }

        self.prediction_history = deque(maxlen=voting_window)

        self.sample_counter = 0
        self.touch_active = False
        self.touch_start_time = None
        self.debounce_time = 0.2

        print(f"Inference engine initialized with {self.model_name} model")
        print(f"Window size: {window_size} samples ({window_size/sample_rate:.1f}s)")
        print(f"Prediction interval: every {prediction_interval} samples")
        print(f"Voting window: {voting_window} predictions")

    def compute_channel_differences(self, eeg_data: Dict[str, float]) -> Dict[str, float]:
        differences = {}
        channel_pairs = {
            'C3-C4': ('C3', 'C4'),
            'P3-P4': ('P3', 'P4'),
            'P7-P8': ('P7', 'P8'),
            'T7-T8': ('T7', 'T8')
        }

        for pair_name, (ch1, ch2) in channel_pairs.items():
            if ch1 in eeg_data and ch2 in eeg_data:
                differences[pair_name] = eeg_data[ch1] - eeg_data[ch2]

        return differences

    def update_buffers(self, sample_data: Dict[str, float]):
        channel_diffs = self.compute_channel_differences(sample_data)

        for pair_name, value in channel_diffs.items():
            if pair_name in self.buffers:
                self.buffers[pair_name].append(value)

        self.sample_counter += 1

    def extract_features_from_buffers(self) -> np.ndarray:
        all_features = {}

        for pair_name, buffer in self.buffers.items():
            if len(buffer) < self.window_size:
                continue

            window_data = np.array(buffer)
            window_features = self.feature_extractor.extract_all_features_for_window(window_data)

            for key, value in window_features.items():
                all_features[f'{pair_name}_{key}'] = value

        feature_values = [all_features.get(f, 0) for f in self.expected_features]
        return np.array(feature_values).reshape(1, -1)

    def predict(self) -> Tuple[int, float]:
        if any(len(buffer) < self.window_size for buffer in self.buffers.values()):
            return 0, 0.0

        features = self.extract_features_from_buffers()
        features_scaled = self.scaler.transform(features)

        prediction = self.model.predict(features_scaled)[0]
        confidence = 0.5

        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = probabilities[1] if prediction == 1 else probabilities[0]

        return int(prediction), float(confidence)

    def apply_voting(self, prediction: int, confidence: float) -> Tuple[bool, float]:
        self.prediction_history.append((prediction, confidence))

        if len(self.prediction_history) < self.voting_window:
            return False, 0.0

        predictions = [p for p, _ in self.prediction_history]
        confidences = [c for _, c in self.prediction_history]

        touch_ratio = sum(predictions) / len(predictions)
        avg_confidence = np.mean(confidences)

        touch_detected = touch_ratio >= 0.6

        return touch_detected, avg_confidence

    def process_sample(self, sample_data: Dict[str, float]) -> Dict[str, any]:
        self.update_buffers(sample_data)

        result = {
            'timestamp': time.time(),
            'touch_detected': self.touch_active,
            'confidence': 0.0,
            'new_prediction': False
        }

        if self.sample_counter % self.prediction_interval == 0:
            prediction, confidence = self.predict()
            touch_detected, avg_confidence = self.apply_voting(prediction, confidence)

            current_time = time.time()

            if touch_detected and not self.touch_active:
                self.touch_active = True
                self.touch_start_time = current_time
                result['event'] = 'touch_start'
            elif not touch_detected and self.touch_active:
                if current_time - self.touch_start_time > self.debounce_time:
                    self.touch_active = False
                    result['event'] = 'touch_end'
                    result['duration'] = current_time - self.touch_start_time

            result['touch_detected'] = self.touch_active
            result['confidence'] = avg_confidence
            result['new_prediction'] = True

        return result

    def process_batch(self, samples: List[Dict[str, float]]) -> List[Dict[str, any]]:
        results = []
        for sample in samples:
            result = self.process_sample(sample)
            results.append(result)
        return results

    def reset(self):
        for buffer in self.buffers.values():
            buffer.clear()
        self.prediction_history.clear()
        self.sample_counter = 0
        self.touch_active = False
        self.touch_start_time = None

    @property
    def expected_features(self) -> List[str]:
        if not hasattr(self, '_expected_features'):
            dummy_features = self.feature_extractor.extract_all_features_for_window(
                np.zeros(self.window_size)
            )
            self._expected_features = []
            for pair in ['C3-C4', 'P3-P4', 'P7-P8', 'T7-T8']:
                for feat in dummy_features.keys():
                    self._expected_features.append(f'{pair}_{feat}')
        return self._expected_features

class StreamSimulator:
    def __init__(self, csv_path: str, sample_rate: int = 250):
        self.sample_rate = sample_rate
        preprocessor = EEGPreprocessor(sample_rate)
        self.df = preprocessor.load_data(csv_path)
        self.current_index = 0

    def get_next_sample(self) -> Optional[Dict[str, float]]:
        if self.current_index >= len(self.df):
            return None

        sample = self.df.iloc[self.current_index].to_dict()
        self.current_index += 1
        return sample

    def get_batch(self, batch_size: int) -> List[Dict[str, float]]:
        samples = []
        for _ in range(batch_size):
            sample = self.get_next_sample()
            if sample is None:
                break
            samples.append(sample)
        return samples

def main():
    print("="*50)
    print("Real-Time EEG Touch Detection Demo")
    print("="*50)

    try:
        inference = RealTimeInference(
            model_path='touch_detection_model.pkl',
            sample_rate=250,
            window_size=250,
            prediction_interval=25,
            voting_window=5
        )
    except FileNotFoundError:
        print("\nModel file not found. Please run train_model.py first.")
        return

    print("\nSimulating real-time stream from CSV data...")
    simulator = StreamSimulator('OpenBCI-RAW-2025-09-13_23-58-04.csv', sample_rate=250)

    print("Processing samples (showing touch events)...\n")
    total_samples = 0
    touch_events = 0
    last_status = False

    batch_size = 100
    max_batches = 100

    for batch_num in range(max_batches):
        samples = simulator.get_batch(batch_size)
        if not samples:
            break

        results = inference.process_batch(samples)

        for result in results:
            total_samples += 1

            if result['new_prediction']:
                if 'event' in result:
                    if result['event'] == 'touch_start':
                        touch_events += 1
                        print(f"[Sample {total_samples}] TOUCH DETECTED! (confidence: {result['confidence']:.2f})")
                    elif result['event'] == 'touch_end':
                        print(f"[Sample {total_samples}] Touch ended (duration: {result['duration']:.2f}s)")

                if total_samples % 1000 == 0:
                    status = "TOUCHING" if result['touch_detected'] else "NO TOUCH"
                    print(f"[Sample {total_samples}] Status: {status} (confidence: {result['confidence']:.2f})")

    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Total samples processed: {total_samples}")
    print(f"Touch events detected: {touch_events}")
    print("="*50)

if __name__ == "__main__":
    main()