import numpy as np
from preprocess import EEGPreprocessor
from features import FeatureExtractor
import joblib

preprocessor = EEGPreprocessor(sample_rate=250)
print("Loading and preprocessing data...")
df_normalized, df_differences = preprocessor.preprocess_pipeline(
    "OpenBCI-RAW-2025-09-13_23-58-04.csv"
)

print(f"\nMarker statistics:")
if 'Marker' in df_normalized.columns:
    markers = df_normalized['Marker'].values
    print(f"Total samples: {len(markers)}")
    print(f"Touch samples (marker > 0): {np.sum(markers > 0)}")
    print(f"Percentage of touch samples: {100 * np.mean(markers > 0):.1f}%")

    first_touch = np.where(markers > 0)[0][0] if np.any(markers > 0) else None
    if first_touch:
        print(f"First touch at sample: {first_touch}")
        print(f"Testing on samples {first_touch-500} to {first_touch+1000}")

        test_region = df_differences.iloc[first_touch-500:first_touch+1000]

        extractor = FeatureExtractor(sample_rate=250, window_size=250)
        features_df, labels = extractor.extract_features_pipeline(test_region, overlap=0.5)

        print(f"\nTest region statistics:")
        print(f"Windows extracted: {len(labels)}")
        print(f"Touch windows: {np.sum(labels == 1)}")
        print(f"No-touch windows: {np.sum(labels == 0)}")

        model_data = joblib.load('touch_detection_model.pkl')
        model = model_data['model']
        scaler = model_data['scaler']

        features_scaled = scaler.transform(features_df)
        predictions = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)[:, 1]

        print(f"\nModel predictions:")
        print(f"Predicted touch windows: {np.sum(predictions == 1)}")
        print(f"Predicted no-touch windows: {np.sum(predictions == 0)}")
        print(f"Accuracy on test region: {np.mean(predictions == labels):.2f}")

        print(f"\nPrediction confidence:")
        print(f"Mean confidence for actual touches: {np.mean(probabilities[labels == 1]):.2f}")
        print(f"Mean confidence for actual no-touches: {1 - np.mean(probabilities[labels == 0]):.2f}")