import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
import warnings
import os
import sys
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from preprocess import EEGPreprocessor
from features import FeatureExtractor

class New1ModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        # Use 250Hz sampling rate for 8-channel setup
        self.preprocessor = EEGPreprocessor(sample_rate=250)
        self.extractor = FeatureExtractor(sample_rate=250, window_size=250)

    def propagate_markers(self, markers: np.ndarray, before_samples: int = 49, after_samples: int = 200) -> np.ndarray:
        """Propagate markers with windows adjusted for 250Hz sampling"""
        propagated = markers.copy()
        marker_indices = np.where(markers > 0)[0]

        for idx in marker_indices:
            start = max(0, idx - before_samples)
            end = min(len(propagated), idx + after_samples + 1)
            propagated[start:end] = markers[idx]

        return propagated

    def process_dataset(self, file_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Process single dataset for binary classification"""
        print(f"\n{'='*50}")
        print(f"Processing {os.path.basename(file_path)}...")
        print('='*50)

        _, df_differences = self.preprocessor.preprocess_pipeline(file_path)

        if 'Marker' in df_differences.columns:
            markers = df_differences['Marker'].values

            # Binary classification: only marker 0 and 1
            # Filter out any marker 2 samples if they exist
            markers = np.where(markers == 2, 0, markers)

            unique, counts = np.unique(markers, return_counts=True)
            marker_dist = dict(zip(unique, counts))
            print(f"Original marker distribution:")
            for m, c in sorted(marker_dist.items()):
                print(f"  Marker {int(m)}: {c:,} samples")

            propagated_markers = self.propagate_markers(markers)
            df_differences['Marker'] = propagated_markers

            unique_prop, counts_prop = np.unique(propagated_markers, return_counts=True)
            marker_dist_prop = dict(zip(unique_prop, counts_prop))
            print(f"\nAfter marker propagation:")
            for m, c in sorted(marker_dist_prop.items()):
                print(f"  Marker {int(m)}: {c:,} samples")

        features_df, labels = self.extractor.extract_features_pipeline(
            df_differences,
            overlap=0.5
        )

        if labels is not None:
            # Binary labels: 0 for no touch, 1 for touch
            binary_labels = np.where(labels > 0, 1, 0)

            touch_windows = np.sum(binary_labels == 1)
            no_touch_windows = np.sum(binary_labels == 0)

            print(f"\nExtracted windows:")
            print(f"  Total: {len(features_df)}")
            print(f"  Touch (marker 1): {touch_windows} ({touch_windows/len(features_df)*100:.1f}%)")
            print(f"  No touch (marker 0): {no_touch_windows} ({no_touch_windows/len(features_df)*100:.1f}%)")

            return features_df, binary_labels
        else:
            return pd.DataFrame(), np.array([])

    def process_multiple_datasets(self, file_paths: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Process and combine multiple datasets"""
        all_features = []
        all_labels = []

        for file_path in file_paths:
            if os.path.exists(file_path):
                features_df, labels = self.process_dataset(file_path)
                if len(features_df) > 0:
                    all_features.append(features_df)
                    all_labels.append(labels)
            else:
                print(f"Warning: File not found - {file_path}")

        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            combined_labels = np.concatenate(all_labels)

            print(f"\n{'='*50}")
            print("COMBINED DATASET SUMMARY")
            print('='*50)
            print(f"Total samples: {len(combined_features)}")
            print(f"Touch samples: {np.sum(combined_labels == 1)} ({np.sum(combined_labels == 1)/len(combined_labels)*100:.1f}%)")
            print(f"No-touch samples: {np.sum(combined_labels == 0)} ({np.sum(combined_labels == 0)/len(combined_labels)*100:.1f}%)")

            return combined_features, combined_labels
        else:
            return pd.DataFrame(), np.array([])

    def train_model(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.3):
        """Train binary classification model optimized for RECALL"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        )

        print(f"\n{'='*50}")
        print("DATA SPLIT")
        print('='*50)
        print(f"Training set: {len(X_train)} samples")
        print(f"  Touch: {np.sum(y_train == 1)}")
        print(f"  No touch: {np.sum(y_train == 0)}")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Handle extreme class imbalance
        if np.sum(y_train == 1) > 1 and np.sum(y_train == 0) > 1:
            print("\nApplying SMOTE for class balancing...")
            # Increased sampling strategy for better recall
            smote = SMOTE(random_state=42, sampling_strategy=0.3)  # Create synthetic samples to get 30% minority class
            try:
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                print(f"After SMOTE:")
                print(f"  Touch: {np.sum(y_train_balanced == 1)}")
                print(f"  No touch: {np.sum(y_train_balanced == 0)}")
            except:
                print("SMOTE failed, using class weights instead")
                X_train_balanced, y_train_balanced = X_train, y_train
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
            print("Skipping SMOTE (insufficient samples)")

        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"\n{'='*50}")
        print("TRAINING RANDOM FOREST MODEL (OPTIMIZED FOR RECALL)")
        print('='*50)

        # Adjusted hyperparameters to maximize recall
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],  # Shallower trees to avoid overfitting
            'min_samples_split': [2, 5],  # Lower values to capture rare events
            'min_samples_leaf': [1, 2],  # Lower values for better recall
            'class_weight': ['balanced', 'balanced_subsample', {0: 1, 1: 10}]  # Heavy weight on minority class
        }

        rf = RandomForestClassifier(random_state=42, n_jobs=-1)

        print("Performing grid search optimized for RECALL...")
        grid_search = GridSearchCV(
            rf, param_grid, cv=3,
            scoring='recall',  # OPTIMIZE FOR RECALL!
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train_scaled, y_train_balanced)

        self.model = grid_search.best_estimator_

        print(f"\nBest parameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")

        # Evaluate on validation set with recall focus
        y_val_pred = self.model.predict(X_val_scaled)
        val_recall = recall_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred, zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred)

        print(f"\nValidation Metrics:")
        print(f"  Recall: {val_recall:.3f} (% of actual touches detected)")
        print(f"  Precision: {val_precision:.3f} (% of predicted touches that are real)")
        print(f"  F1-Score: {val_f1:.3f}")

        return X_test_scaled, y_test

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate model performance on test set with focus on recall"""
        print(f"\n{'='*50}")
        print("MODEL EVALUATION (RECALL-FOCUSED)")
        print('='*50)

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate all metrics
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred)

        print(f"Test Performance:")
        print(f"  RECALL: {recall:.3f} ← PRIMARY METRIC")
        print(f"  Precision: {precision:.3f}")
        print(f"  F1-Score: {f1:.3f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                   target_names=['No Touch', 'Touch'],
                                   zero_division=0))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"True Negatives: {cm[0, 0] if cm.shape[0] > 0 else 0}")
        print(f"False Positives: {cm[0, 1] if cm.shape[1] > 1 else 0}")
        print(f"False Negatives: {cm[1, 0] if cm.shape[0] > 1 else 0}")
        print(f"True Positives: {cm[1, 1] if cm.shape == (2, 2) else 0}")

        if cm.shape == (2, 2):
            print(f"\nDetection Rate: {cm[1, 1]}/{cm[1, 0] + cm[1, 1]} touches detected")

        self.plot_results(y_test, y_pred, y_proba, recall, precision, f1)

        return recall, precision, f1

    def plot_results(self, y_test, y_pred, y_proba, recall, precision, f1):
        """Generate performance visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Touch', 'Touch'],
                   yticklabels=['No Touch', 'Touch'], ax=axes[0, 0])
        axes[0, 0].set_title(f'Confusion Matrix (Recall: {recall:.3f})')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')

        # Metrics Bar Chart
        metrics = ['Recall', 'Precision', 'F1-Score']
        scores = [recall, precision, f1]
        colors = ['green' if s > 0.5 else 'orange' if s > 0.3 else 'red' for s in scores]
        axes[0, 1].bar(metrics, scores, color=colors, alpha=0.7)
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Performance Metrics (Recall-Optimized)')
        axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        for i, (m, s) in enumerate(zip(metrics, scores)):
            axes[0, 1].text(i, s + 0.02, f'{s:.3f}', ha='center')

        # Feature Importance
        feature_importance = self.model.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-20:]
        top_features = feature_importance[top_features_idx]

        axes[1, 0].barh(range(len(top_features)), top_features, color='green', alpha=0.7)
        axes[1, 0].set_xlabel('Feature Importance')
        axes[1, 0].set_title('Top 20 Most Important Features')
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels([f'Feature {i}' for i in top_features_idx])

        # Probability Distribution
        axes[1, 1].hist(y_proba[y_test == 0], bins=30, alpha=0.5, label='No Touch', color='blue')
        axes[1, 1].hist(y_proba[y_test == 1], bins=30, alpha=0.5, label='Touch', color='red')
        axes[1, 1].set_xlabel('Predicted Probability of Touch')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Probability Distribution by Class')
        axes[1, 1].legend()
        axes[1, 1].axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Default Threshold')

        plt.tight_layout()

        # Create model directory if it doesn't exist
        os.makedirs('../models/new1_model', exist_ok=True)
        plt.savefig('../models/new1_model/performance_plot.png', dpi=100, bbox_inches='tight')
        plt.show()
        print("\nPerformance plot saved as 'performance_plot.png'")

    def save_model(self, recall: float, precision: float, f1: float, dataset_paths: List[str]):
        """Save trained model and metadata"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'recall': recall,
            'precision': precision,
            'f1_score': f1,
            'sample_rate': 250,
            'window_size': 250
        }

        # Create model directory if it doesn't exist
        os.makedirs('../models/new1_model', exist_ok=True)

        model_path = '../models/new1_model/new1_model.pkl'
        joblib.dump(model_data, model_path)
        print(f"\nModel saved to: {model_path}")

        info_path = '../models/new1_model/model_info.txt'
        with open(info_path, 'w') as f:
            f.write("MODEL: new1_model\n")
            f.write("="*50 + "\n")
            f.write("Binary Classification Model for Touch Detection (Recall-Optimized)\n\n")
            f.write(f"Training Date: 2025-09-14\n")
            f.write(f"Model Type: Random Forest Classifier\n")
            f.write(f"Task: Binary classification (Touch vs No Touch)\n")
            f.write(f"Optimization: RECALL (Detect all touches)\n\n")

            f.write("CLASS MAPPING:\n")
            f.write("- Class 0: No Touch (marker = 0)\n")
            f.write("- Class 1: Touch (marker = 1)\n\n")

            f.write("PERFORMANCE METRICS:\n")
            f.write(f"- Test RECALL: {recall*100:.1f}% ← PRIMARY METRIC\n")
            f.write(f"- Test Precision: {precision*100:.1f}%\n")
            f.write(f"- Test F1-Score: {f1*100:.1f}%\n\n")

            f.write(f"TRAINING DATA:\n")
            for path in dataset_paths:
                f.write(f"- {os.path.basename(path)}\n")
            f.write("\n")

            f.write("PREPROCESSING:\n")
            f.write("- Sample Rate: 250 Hz\n")
            f.write("- Window Size: 250 samples (1 second)\n")
            f.write("- Overlap: 50%\n")
            f.write("- Marker Propagation: 49 samples before, 200 samples after\n")
            f.write("- Class Balancing: SMOTE (30% minority) + class weights\n\n")

            f.write("FEATURE CONFIGURATION:\n")
            f.write("- Channel Pairs: C3-C4, P3-P4, P7-P8, T7-T8\n")
            f.write("- Frequency Bands: Delta, Theta, Alpha, Beta, Gamma\n")
            f.write("- Total Features: 160\n\n")

            f.write("HYPERPARAMETERS (Best):\n")
            if self.model:
                for param, value in self.model.get_params().items():
                    if param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'class_weight']:
                        f.write(f"- {param}: {value}\n")

            f.write("\nFILES:\n")
            f.write("- new1_model.pkl: Trained model with scaler\n")
            f.write("- performance_plot.png: Performance visualization\n")
            f.write("- model_info.txt: This configuration file\n")

        print(f"Model info saved to: {info_path}")

def main():
    print("="*60)
    print("NEW1 MODEL TRAINING - RECALL-OPTIMIZED BINARY CLASSIFICATION")
    print("Training on multiple datasets at 250Hz")
    print("="*60)

    trainer = New1ModelTrainer()

    # List of datasets to train on
    data_files = [
        '../datasets/daniel_updated.csv',
        '../datasets/OpenBCI-RAW-2025-09-14_09-02-43.txt'
    ]

    # Process and combine all datasets
    features_df, labels = trainer.process_multiple_datasets(data_files)

    if len(features_df) == 0:
        print("Error: No data extracted from files")
        return

    print(f"\n{'='*50}")
    print("FINAL DATASET SUMMARY")
    print('='*50)
    print(f"Total windows: {len(features_df)}")
    print(f"Feature dimensions: {features_df.shape}")
    print(f"Touch windows: {np.sum(labels == 1)}")
    print(f"No touch windows: {np.sum(labels == 0)}")

    # Warning about imbalance
    if np.sum(labels == 1) < 50:
        print("\n⚠️  WARNING: Low number of positive samples!")
        print("Model is optimized for RECALL to detect all touches.")

    X_test, y_test = trainer.train_model(features_df.values, labels)

    recall, precision, f1 = trainer.evaluate_model(X_test, y_test)

    trainer.save_model(recall, precision, f1, data_files)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Final Test RECALL: {recall*100:.1f}% (of actual touches detected)")
    print(f"Final Test Precision: {precision*100:.1f}%")
    print(f"Final Test F1-Score: {f1*100:.1f}%")
    print("="*60)

if __name__ == "__main__":
    main()