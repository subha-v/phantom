#!/usr/bin/env python3
"""Test script to debug inference issues"""

import sys
import os
import numpy as np
import joblib

# Add model_training utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model_training', 'utils'))

from features import FeatureExtractor

def test_model_loading(model_name="new_model"):
    """Test if model loads correctly"""
    model_path = f"../model_training/models/{model_name}/{model_name}.pkl" if model_name == "new_model" else f"../model_training/models/{model_name}/{model_name}_model.pkl"

    print("Testing model loading...")
    print(f"Model name: {model_name}")
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

def test_feature_extraction(sample_rate=250):
    """Test feature extraction"""
    print("\nTesting feature extraction...")
    print(f"Sample rate: {sample_rate}Hz")

    # Create dummy data (8 channels, samples for 1 second)
    window_size = sample_rate  # 1 second of data
    dummy_window = np.random.randn(8, window_size)

    # Create extractor
    extractor = FeatureExtractor(sample_rate=sample_rate, window_size=window_size)

    # Compute channel differences
    differences = np.zeros((4, window_size))
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

def test_synthetic_accuracy(model, scaler, num_samples=100, sample_rate=250):
    """Test model accuracy on synthetic random data"""
    print("\n" + "="*60)
    print("TESTING ACCURACY ON SYNTHETIC DATA")
    print("="*60)
    print(f"Generating {num_samples} synthetic samples...")

    predictions = []
    confidences = []

    # Create feature extractor
    window_size = sample_rate
    extractor = FeatureExtractor(sample_rate=sample_rate, window_size=window_size)

    for i in range(num_samples):
        # Generate random synthetic EEG data
        dummy_window = np.random.randn(8, window_size)

        # Compute channel differences
        differences = np.zeros((4, window_size))
        differences[0] = dummy_window[2] - dummy_window[3]  # C3-C4
        differences[1] = dummy_window[6] - dummy_window[7]  # P3-P4
        differences[2] = dummy_window[4] - dummy_window[5]  # P7-P8
        differences[3] = dummy_window[0] - dummy_window[1]  # T7-T8

        # Extract features
        channel_pairs = ['C3-C4', 'P3-P4', 'P7-P8', 'T7-T8']
        all_features_dict = {}

        for pair_name, channel_diff in zip(channel_pairs, differences):
            features = extractor.extract_all_features_for_window(channel_diff)
            for key, value in features.items():
                all_features_dict[f'{pair_name}_{key}'] = value

        # Convert to array
        feature_array = np.array(list(all_features_dict.values()), dtype=np.float32)
        feature_array = feature_array.reshape(1, -1)

        # Scale features
        if scaler:
            features_scaled = scaler.transform(feature_array)
        else:
            features_scaled = feature_array

        # Make prediction
        try:
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]

            predictions.append(prediction)
            confidences.append(probabilities[1] if len(probabilities) > 1 else 0)
        except Exception as e:
            print(f"Error during inference for sample {i}: {e}")
            continue

    # Calculate statistics
    predictions = np.array(predictions)
    confidences = np.array(confidences)

    touch_predictions = np.sum(predictions == 1)
    no_touch_predictions = np.sum(predictions == 0)
    touch_percentage = (touch_predictions / len(predictions)) * 100

    print(f"\nResults on {len(predictions)} synthetic samples:")
    print(f"  Touch predictions: {touch_predictions} ({touch_percentage:.1f}%)")
    print(f"  No-touch predictions: {no_touch_predictions} ({100-touch_percentage:.1f}%)")
    print(f"  Average confidence for touch: {np.mean(confidences):.3f}")
    print(f"  Max confidence: {np.max(confidences):.3f}")
    print(f"  Min confidence: {np.min(confidences):.3f}")

    print("\nNOTE: Since this is synthetic random data, we expect:")
    print("  - Low confidence scores")
    print("  - Mostly 'no-touch' predictions")
    print("  - This tests that the model doesn't have false positives on noise")

    return predictions, confidences

def test_inference(model, scaler, features):
    """Test model inference on single sample"""
    print("\nTesting single sample inference...")

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

    # Use the new_model
    model_name = "new_model"
    print(f"\nUsing model: {model_name}")

    # Test model loading
    model, scaler = test_model_loading(model_name)

    if model is None:
        print("Failed to load model, exiting...")
        return

    # Detect sample rate from model
    sample_rate = 250  # Default
    if hasattr(model, 'n_features_in_'):
        # The model expects 160 features, this is consistent with 250Hz
        # (would be different for 500Hz)
        print(f"Model expects {model.n_features_in_} features")
        if model.n_features_in_ == 160:
            print("This is consistent with 250Hz sampling rate")
            sample_rate = 250

    # Test feature extraction
    features = test_feature_extraction(sample_rate)

    # Test single inference
    test_inference(model, scaler, features)

    # Test accuracy on synthetic data
    test_synthetic_accuracy(model, scaler, num_samples=100, sample_rate=sample_rate)

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()