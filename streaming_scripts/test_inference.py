#!/usr/bin/env python3
"""Test script to debug inference issues"""

import sys
import os
import numpy as np
import joblib

# Add model_training utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model_training', 'utils'))

from features import FeatureExtractor

def test_model_loading():
    """Test if model loads correctly"""
    model_path = "../model_training/models/marker1_only_4data/marker1_only_4data_model.pkl"

    print("Testing model loading...")
    print(f"Model path: {model_path}")

    # Check if file exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return None, None

    print(f"File exists, size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

    try:
        # Load model
        model_data = joblib.load(model_path)
        print("Model loaded successfully!")

        if isinstance(model_data, dict):
            model = model_data.get('model')
            scaler = model_data.get('scaler')
            print(f"Model type: {type(model).__name__}")
            print(f"Scaler type: {type(scaler).__name__}")
            print(f"Accuracy: {model_data.get('accuracy', 'N/A'):.1%}")

            # Check feature count
            if hasattr(model, 'n_features_in_'):
                print(f"Expected features: {model.n_features_in_}")

            return model, scaler
        else:
            print("Model data is not a dictionary")
            return model_data, None

    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None, None

def test_feature_extraction():
    """Test feature extraction"""
    print("\nTesting feature extraction...")

    # Create dummy data (8 channels, 250 samples at 250Hz = 1 second)
    dummy_window = np.random.randn(8, 250)

    # Create extractor
    extractor = FeatureExtractor(sample_rate=250, window_size=250)

    # Compute channel differences
    differences = np.zeros((4, 250))
    differences[0] = dummy_window[2] - dummy_window[3]  # C3-C4
    differences[1] = dummy_window[6] - dummy_window[7]  # P3-P4
    differences[2] = dummy_window[4] - dummy_window[5]  # P7-P8
    differences[3] = dummy_window[0] - dummy_window[1]  # T7-T8

    # Extract features
    channel_pairs = ['C3-C4', 'P3-P4', 'P7-P8', 'T7-T8']
    all_features_dict = {}

    for pair_name, channel_diff in zip(channel_pairs, differences):
        features = extractor.extract_all_features_for_window(channel_diff)
        print(f"Features for {pair_name}: {len(features)} features")

        # Add prefix
        for key, value in features.items():
            all_features_dict[f'{pair_name}_{key}'] = value

    print(f"Total features extracted: {len(all_features_dict)}")

    # Convert to array
    feature_array = np.array(list(all_features_dict.values()), dtype=np.float32)
    print(f"Feature array shape: {feature_array.shape}")

    return feature_array

def test_inference(model, scaler, features):
    """Test model inference"""
    print("\nTesting inference...")

    if model is None:
        print("No model loaded, skipping inference")
        return

    # Reshape for single sample
    features = features.reshape(1, -1)

    # Scale features
    if scaler:
        features_scaled = scaler.transform(features)
        print("Features scaled")
    else:
        features_scaled = features
        print("No scaler, using raw features")

    # Make prediction
    try:
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        print(f"Prediction: {prediction}")
        print(f"Probabilities: {probabilities}")
        print(f"Confidence for marker 1: {probabilities[1] if len(probabilities) > 1 else 0:.2%}")

    except Exception as e:
        print(f"ERROR during inference: {e}")

def main():
    print("="*60)
    print("INFERENCE TEST SCRIPT")
    print("="*60)

    # Test model loading
    model, scaler = test_model_loading()

    # Test feature extraction
    features = test_feature_extraction()

    # Test inference
    test_inference(model, scaler, features)

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()