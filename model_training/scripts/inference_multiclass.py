import numpy as np
import joblib
from collections import deque
from typing import Dict, Tuple, Optional, List
import time
from preprocess import EEGPreprocessor
from features import FeatureExtractor

class MulticlassRealTimeInference:
    """
    Real-time inference for rubber hand illusion multiclass detection

    Classes:
    - 0: No Touch
    - 1: Both Hands (real + fake)
    - 2: Fake Hand Only
    """

    def __init__(self, model_path: str = 'multiclass_model.pkl',
                 sample_rate: int = 250, window_size: int = 250,
                 prediction_interval: int = 10, voting_window: int = 5):
        """
        Initialize multiclass inference engine

        Args:
            model_path: Path to trained multiclass model
            sample_rate: EEG sampling rate in Hz
            window_size: Window size in samples
            prediction_interval: Predict every N samples
            voting_window: Number of predictions for voting
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.prediction_interval = prediction_interval
        self.voting_window = voting_window

        # Load model and configuration
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_name = model_data.get('model_name', 'Unknown')
        self.class_names = model_data.get('class_names', ['No Touch', 'Both Hands', 'Fake Hand Only'])
        self.n_classes = model_data.get('n_classes', 3)

        # Initialize preprocessor and feature extractor
        self.preprocessor = EEGPreprocessor(sample_rate)
        self.feature_extractor = FeatureExtractor(sample_rate, window_size)

        # Initialize buffers for each channel pair
        self.buffers = {
            'C3-C4': deque(maxlen=window_size),
            'P3-P4': deque(maxlen=window_size),
            'P7-P8': deque(maxlen=window_size),
            'T7-T8': deque(maxlen=window_size)
        }

        # Prediction history for voting
        self.prediction_history = deque(maxlen=voting_window)
        self.confidence_history = deque(maxlen=voting_window)

        # State tracking
        self.sample_counter = 0
        self.current_state = 0  # Default to "No Touch"
        self.state_start_time = None
        self.state_durations = {0: [], 1: [], 2: []}
        self.transition_matrix = np.zeros((self.n_classes, self.n_classes))

        # Confidence thresholds for each class
        self.confidence_thresholds = {
            0: 0.4,  # No Touch
            1: 0.5,  # Both Hands
            2: 0.4   # Fake Hand Only
        }

        print(f"Multiclass Inference Engine Initialized")
        print(f"Model: {self.model_name}")
        print(f"Classes: {', '.join(self.class_names)}")
        print(f"Window size: {window_size} samples ({window_size/sample_rate:.1f}s)")
        print(f"Prediction interval: every {prediction_interval} samples")

    def compute_channel_differences(self, eeg_data: Dict[str, float]) -> Dict[str, float]:
        """Compute symmetric channel differences"""
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
        """Update circular buffers with new sample"""
        channel_diffs = self.compute_channel_differences(sample_data)

        for pair_name, value in channel_diffs.items():
            if pair_name in self.buffers:
                self.buffers[pair_name].append(value)

        self.sample_counter += 1

    def extract_features_from_buffers(self) -> np.ndarray:
        """Extract features from current buffer state"""
        all_features = {}

        for pair_name, buffer in self.buffers.items():
            if len(buffer) < self.window_size:
                continue

            window_data = np.array(buffer)
            window_features = self.feature_extractor.extract_all_features_for_window(window_data)

            # Prefix with channel pair name
            for key, value in window_features.items():
                all_features[f'{pair_name}_{key}'] = value

        # Ensure feature order matches training
        feature_values = [all_features.get(f, 0) for f in self.expected_features]
        return np.array(feature_values).reshape(1, -1)

    def predict(self) -> Tuple[int, np.ndarray]:
        """Make multiclass prediction"""
        if any(len(buffer) < self.window_size for buffer in self.buffers.values()):
            return 0, np.array([1.0, 0.0, 0.0])  # Default to "No Touch"

        # Extract and scale features
        features = self.extract_features_from_buffers()
        features_scaled = self.scaler.transform(features)

        # Get prediction and probabilities
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0] if hasattr(self.model, 'predict_proba') else None

        if probabilities is None:
            # Create dummy probabilities if not available
            probabilities = np.zeros(self.n_classes)
            probabilities[prediction] = 1.0

        return int(prediction), probabilities

    def apply_voting_and_smoothing(self, prediction: int, probabilities: np.ndarray) -> Tuple[int, float, str]:
        """Apply voting and state smoothing for stable predictions"""
        self.prediction_history.append(prediction)
        self.confidence_history.append(probabilities)

        if len(self.prediction_history) < self.voting_window:
            return self.current_state, probabilities[self.current_state], "buffering"

        # Calculate weighted vote based on confidence
        vote_weights = np.zeros(self.n_classes)
        for pred, conf in zip(self.prediction_history, self.confidence_history):
            vote_weights += conf

        vote_weights /= len(self.prediction_history)
        voted_class = np.argmax(vote_weights)
        confidence = vote_weights[voted_class]

        # State transition logic with hysteresis
        transition_type = "stable"

        if voted_class != self.current_state:
            # Check if confidence exceeds threshold for transition
            if confidence > self.confidence_thresholds[voted_class]:
                # Record state transition
                current_time = time.time()
                if self.state_start_time is not None:
                    duration = current_time - self.state_start_time
                    self.state_durations[self.current_state].append(duration)

                # Update transition matrix
                self.transition_matrix[self.current_state, voted_class] += 1

                # Change state
                old_state = self.current_state
                self.current_state = voted_class
                self.state_start_time = current_time
                transition_type = f"transition_{self.class_names[old_state]}_to_{self.class_names[voted_class]}"

        return self.current_state, confidence, transition_type

    def process_sample(self, sample_data: Dict[str, float]) -> Dict[str, any]:
        """Process single EEG sample"""
        self.update_buffers(sample_data)

        result = {
            'timestamp': time.time(),
            'state': self.current_state,
            'state_name': self.class_names[self.current_state],
            'confidence': 0.0,
            'new_prediction': False,
            'probabilities': None
        }

        # Make prediction at specified intervals
        if self.sample_counter % self.prediction_interval == 0:
            prediction, probabilities = self.predict()
            state, confidence, transition_type = self.apply_voting_and_smoothing(prediction, probabilities)

            result['state'] = state
            result['state_name'] = self.class_names[state]
            result['confidence'] = confidence
            result['new_prediction'] = True
            result['probabilities'] = probabilities.tolist()
            result['transition'] = transition_type

            # Add detailed probability breakdown
            result['class_probabilities'] = {
                name: float(prob) for name, prob in zip(self.class_names, probabilities)
            }

        return result

    def get_statistics(self) -> Dict[str, any]:
        """Get inference statistics"""
        stats = {
            'total_samples': self.sample_counter,
            'predictions_made': self.sample_counter // self.prediction_interval,
            'current_state': self.class_names[self.current_state],
            'state_durations': {}
        }

        # Calculate average state durations
        for state, durations in self.state_durations.items():
            if durations:
                stats['state_durations'][self.class_names[state]] = {
                    'mean': np.mean(durations),
                    'std': np.std(durations),
                    'count': len(durations)
                }

        # Normalize transition matrix
        if np.sum(self.transition_matrix) > 0:
            stats['transition_probabilities'] = (
                self.transition_matrix / np.sum(self.transition_matrix, axis=1, keepdims=True)
            ).tolist()

        return stats

    def reset(self):
        """Reset inference engine state"""
        for buffer in self.buffers.values():
            buffer.clear()
        self.prediction_history.clear()
        self.confidence_history.clear()
        self.sample_counter = 0
        self.current_state = 0
        self.state_start_time = None

    @property
    def expected_features(self) -> List[str]:
        """Get expected feature names in correct order"""
        if not hasattr(self, '_expected_features'):
            dummy_features = self.feature_extractor.extract_all_features_for_window(
                np.zeros(self.window_size)
            )
            self._expected_features = []
            for pair in ['C3-C4', 'P3-P4', 'P7-P8', 'T7-T8']:
                for feat in dummy_features.keys():
                    self._expected_features.append(f'{pair}_{feat}')
        return self._expected_features


def demo_multiclass_inference():
    """Demo multiclass inference with simulated data"""
    print("="*60)
    print("Multiclass Rubber Hand Illusion Detection Demo")
    print("="*60)

    # Initialize inference engine
    try:
        inference = MulticlassRealTimeInference(
            model_path='multiclass_model.pkl',
            sample_rate=250,
            window_size=250,
            prediction_interval=25,
            voting_window=5
        )
    except FileNotFoundError:
        print("\nModel file not found. Please run train_model_multiclass.py first.")
        return

    # Simulate data stream
    print("\nSimulating real-time EEG stream...")
    from inference import StreamSimulator
    simulator = StreamSimulator('OpenBCI-RAW-2025-09-13_23-58-04.csv', sample_rate=250)

    print("Processing samples...\n")

    # Track state changes
    state_changes = []
    batch_size = 100
    max_batches = 50

    for batch_num in range(max_batches):
        samples = []
        for _ in range(batch_size):
            sample = simulator.get_next_sample()
            if sample is None:
                break
            samples.append(sample)

        if not samples:
            break

        # Process batch
        for sample in samples:
            result = inference.process_sample(sample)

            if result['new_prediction'] and 'transition' in result:
                if result['transition'].startswith('transition_'):
                    state_changes.append({
                        'sample': inference.sample_counter,
                        'transition': result['transition'],
                        'confidence': result['confidence']
                    })
                    print(f"[Sample {inference.sample_counter}] STATE CHANGE: {result['transition']}")
                    print(f"  Confidence: {result['confidence']:.2f}")
                    if result['probabilities']:
                        print(f"  Class probabilities: {result['class_probabilities']}")

            # Periodic status update
            if inference.sample_counter % 2500 == 0:
                print(f"[Sample {inference.sample_counter}] Current: {result['state_name']} "
                      f"(confidence: {result['confidence']:.2f})")

    # Print final statistics
    print(f"\n{'='*60}")
    print("Inference Statistics:")
    print("="*60)

    stats = inference.get_statistics()
    print(f"Total samples processed: {stats['total_samples']}")
    print(f"Total predictions made: {stats['predictions_made']}")
    print(f"Final state: {stats['current_state']}")
    print(f"Total state changes: {len(state_changes)}")

    if stats['state_durations']:
        print("\nAverage state durations:")
        for state, duration_stats in stats['state_durations'].items():
            print(f"  {state}: {duration_stats['mean']:.2f}s ± {duration_stats['std']:.2f}s "
                  f"(n={duration_stats['count']})")

    print("="*60)


if __name__ == "__main__":
    demo_multiclass_inference()