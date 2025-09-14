import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

import numpy as np
import pandas as pd
import joblib
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from preprocess import EEGPreprocessor
from features import FeatureExtractor

class TouchDetectionInference:
    def __init__(self, model_name: str = '3data_marker12_or_none'):
        self.model_name = model_name
        self.model_path = f'../models/{model_name}/{model_name}_model.pkl'
        self.model = None
        self.scaler = None
        self.preprocessor = EEGPreprocessor(sample_rate=125)
        self.extractor = FeatureExtractor(sample_rate=125, window_size=125)

    def load_model(self):
        print(f"Loading model: {self.model_name}")
        model_data = joblib.load(self.model_path)

        if isinstance(model_data, dict):
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler')
        else:
            self.model = model_data
            print("Warning: No scaler found in model file")

        print(f"Model loaded successfully")
        return True

    def predict_from_file(self, csv_file: str) -> Tuple[np.ndarray, np.ndarray]:
        print(f"\nProcessing file: {csv_file}")

        _, df_differences = self.preprocessor.preprocess_pipeline(csv_file)

        features_df, original_labels = self.extractor.extract_features_pipeline(
            df_differences,
            overlap=0.5
        )

        if len(features_df) == 0:
            print("No features extracted")
            return np.array([]), np.array([])

        X_scaled = self.scaler.transform(features_df) if self.scaler else features_df

        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        return predictions, probabilities

    def predict_realtime(self, eeg_window: np.ndarray) -> Tuple[int, float]:
        if len(eeg_window) < self.extractor.window_size:
            return 0, 0.0

        features = self.extractor.extract_all_features_for_window(eeg_window)

        if self.scaler:
            features_scaled = self.scaler.transform([features])
        else:
            features_scaled = [features]

        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]

        return prediction, probability[1]

    def evaluate_predictions(self, predictions: np.ndarray, true_labels: Optional[np.ndarray] = None):
        touch_count = np.sum(predictions == 1)
        no_touch_count = np.sum(predictions == 0)
        total = len(predictions)

        print("\nPrediction Summary:")
        print(f"  Total windows: {total}")
        print(f"  Touch detected: {touch_count} ({touch_count/total*100:.1f}%)")
        print(f"  No touch: {no_touch_count} ({no_touch_count/total*100:.1f}%)")

        if true_labels is not None and len(true_labels) == len(predictions):
            binary_labels = np.where((true_labels == 1) | (true_labels == 2), 1, 0)
            accuracy = np.mean(predictions == binary_labels)
            print(f"\n  Accuracy vs true labels: {accuracy*100:.1f}%")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Touch Detection Inference')
    parser.add_argument('--model', default='3data_marker12_or_none',
                       choices=['marker1_or_not', 'marker12_or_none', 'multiclass_marker', '3data_marker12_or_none'],
                       help='Model to use for inference')
    parser.add_argument('--file', required=True, help='CSV file to process')
    args = parser.parse_args()

    inference = TouchDetectionInference(model_name=args.model)

    if not inference.load_model():
        print("Failed to load model")
        return

    predictions, probabilities = inference.predict_from_file(args.file)

    if len(predictions) > 0:
        inference.evaluate_predictions(predictions)

        print("\nFirst 10 predictions:")
        for i in range(min(10, len(predictions))):
            label = "Touch" if predictions[i] == 1 else "No Touch"
            confidence = probabilities[i][1] if predictions[i] == 1 else probabilities[i][0]
            print(f"  Window {i+1}: {label} (confidence: {confidence:.2f})")

if __name__ == "__main__":
    main()