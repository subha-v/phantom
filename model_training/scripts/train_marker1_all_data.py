import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
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

class Marker1Trainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.preprocessor = EEGPreprocessor(sample_rate=125)
        self.extractor = FeatureExtractor(sample_rate=125, window_size=125)

    def propagate_markers(self, markers: np.ndarray, before_samples: int = 24, after_samples: int = 100) -> np.ndarray:
        propagated = markers.copy()
        marker_indices = np.where(markers > 0)[0]

        for idx in marker_indices:
            start = max(0, idx - before_samples)
            end = min(len(propagated), idx + after_samples + 1)
            propagated[start:end] = markers[idx]

        return propagated

    def process_datasets(self, file_paths: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
        all_features = []
        all_labels = []

        dataset_stats = {}

        for file_path in file_paths:
            filename = os.path.basename(file_path)
            print(f"\n{'='*50}")
            print(f"Processing {filename}...")
            print('='*50)

            _, df_differences = self.preprocessor.preprocess_pipeline(file_path)

            if 'Marker' in df_differences.columns:
                markers = df_differences['Marker'].values

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
                binary_labels = np.where(labels == 1, 1, 0)

                marker1_windows = np.sum(binary_labels == 1)
                not_marker1_windows = np.sum(binary_labels == 0)

                all_features.append(features_df)
                all_labels.append(binary_labels)

                print(f"\nExtracted windows:")
                print(f"  Total: {len(features_df)}")
                print(f"  Marker 1: {marker1_windows} ({marker1_windows/len(features_df)*100:.1f}%)")
                print(f"  Not Marker 1: {not_marker1_windows} ({not_marker1_windows/len(features_df)*100:.1f}%)")

                dataset_stats[filename] = {
                    'total_windows': len(features_df),
                    'marker1': marker1_windows,
                    'not_marker1': not_marker1_windows
                }

        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            combined_labels = np.concatenate(all_labels)

            print(f"\n{'='*50}")
            print("COMBINED DATASET SUMMARY")
            print('='*50)
            print(f"Total windows: {len(combined_features)}")
            print(f"Feature dimensions: {combined_features.shape}")
            print(f"Marker 1 windows: {np.sum(combined_labels == 1)} ({np.sum(combined_labels == 1)/len(combined_labels)*100:.1f}%)")
            print(f"Not Marker 1 windows: {np.sum(combined_labels == 0)} ({np.sum(combined_labels == 0)/len(combined_labels)*100:.1f}%)")

            return combined_features, combined_labels, dataset_stats
        else:
            return pd.DataFrame(), np.array([]), {}

    def train_model(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.3):
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
        print(f"  Marker 1: {np.sum(y_train == 1)}")
        print(f"  Not Marker 1: {np.sum(y_train == 0)}")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")

        if np.sum(y_train == 1) > 1 and np.sum(y_train == 0) > 1:
            print("\nApplying SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE:")
            print(f"  Marker 1: {np.sum(y_train_balanced == 1)}")
            print(f"  Not Marker 1: {np.sum(y_train_balanced == 0)}")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
            print("Skipping SMOTE (insufficient samples)")

        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"\n{'='*50}")
        print("TRAINING RANDOM FOREST MODEL")
        print('='*50)

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        rf = RandomForestClassifier(random_state=42, n_jobs=-1)

        print("Performing grid search for best hyperparameters...")
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train_scaled, y_train_balanced)

        self.model = grid_search.best_estimator_

        print(f"\nBest parameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")

        val_score = self.model.score(X_val_scaled, y_val)
        print(f"\nValidation accuracy: {val_score:.3f}")

        return X_test_scaled, y_test

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray):
        print(f"\n{'='*50}")
        print("MODEL EVALUATION")
        print('='*50)

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        print(f"Test Accuracy: {accuracy:.3f}")
        print(f"ROC AUC: {roc_auc:.3f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                   target_names=['Not Marker 1', 'Marker 1']))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"True Negatives: {cm[0, 0]}")
        print(f"False Positives: {cm[0, 1]}")
        print(f"False Negatives: {cm[1, 0]}")
        print(f"True Positives: {cm[1, 1]}")

        self.plot_results(y_test, y_pred, y_proba, accuracy, roc_auc)

        return accuracy, roc_auc

    def plot_results(self, y_test, y_pred, y_proba, accuracy, roc_auc):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Not Marker 1', 'Marker 1'],
                   yticklabels=['Not Marker 1', 'Marker 1'], ax=axes[0, 0])
        axes[0, 0].set_title(f'Confusion Matrix (Acc: {accuracy:.3f})')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')

        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        axes[0, 1].plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True, alpha=0.3)

        feature_importance = self.model.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-20:]
        top_features = feature_importance[top_features_idx]

        axes[1, 0].barh(range(len(top_features)), top_features, color='green', alpha=0.7)
        axes[1, 0].set_xlabel('Feature Importance')
        axes[1, 0].set_title('Top 20 Most Important Features')
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels([f'Feature {i}' for i in top_features_idx])

        from sklearn.calibration import calibration_curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_proba, n_bins=10
        )
        axes[1, 1].plot(mean_predicted_value, fraction_of_positives, 's-', color='blue', label='Model')
        axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        axes[1, 1].set_xlabel('Mean predicted probability')
        axes[1, 1].set_ylabel('Fraction of positives')
        axes[1, 1].set_title('Calibration Plot')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('../models/marker1_only_4data/performance_plot.png', dpi=100, bbox_inches='tight')
        plt.show()
        print("\nPerformance plot saved as 'performance_plot.png'")

    def save_model(self, accuracy: float, roc_auc: float, dataset_stats: dict):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'dataset_stats': dataset_stats
        }

        model_path = '../models/marker1_only_4data/marker1_only_4data_model.pkl'
        joblib.dump(model_data, model_path)
        print(f"\nModel saved to: {model_path}")

        info_path = '../models/marker1_only_4data/model_info.txt'
        with open(info_path, 'w') as f:
            f.write("MODEL: marker1_only_4data\n")
            f.write("="*50 + "\n")
            f.write("Binary Classification Model for Marker 1 Detection\n\n")
            f.write(f"Training Date: 2025-09-14\n")
            f.write(f"Model Type: Random Forest Classifier\n")
            f.write(f"Task: Binary classification (Marker 1 vs Not Marker 1)\n\n")

            f.write("CLASS MAPPING:\n")
            f.write("- Class 0: Not Marker 1 (includes marker = 0 AND marker = 2)\n")
            f.write("- Class 1: Marker 1 (marker = 1 only)\n\n")

            f.write("PERFORMANCE METRICS:\n")
            f.write(f"- Test Accuracy: {accuracy*100:.1f}%\n")
            f.write(f"- ROC AUC: {roc_auc:.3f}\n\n")

            f.write("TRAINING DATA (4 datasets):\n")
            for dataset, stats in dataset_stats.items():
                f.write(f"- {dataset}:\n")
                f.write(f"    Total windows: {stats['total_windows']}\n")
                f.write(f"    Marker 1: {stats['marker1']}\n")
                f.write(f"    Not Marker 1: {stats['not_marker1']}\n")

            f.write("\nPREPROCESSING:\n")
            f.write("- Sample Rate: 125 Hz\n")
            f.write("- Window Size: 125 samples (1 second)\n")
            f.write("- Overlap: 50%\n")
            f.write("- Marker Propagation: 24 samples before, 100 samples after\n")
            f.write("- Class Balancing: SMOTE\n\n")

            f.write("FEATURE CONFIGURATION:\n")
            f.write("- Channel Pairs: C3-C4, P3-P4, P7-P8, T7-T8\n")
            f.write("- Frequency Bands: Delta, Theta, Alpha, Beta, Gamma\n")
            f.write("- Total Features: 160\n\n")

            f.write("HYPERPARAMETERS (Best):\n")
            for param, value in self.model.get_params().items():
                if param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
                    f.write(f"- {param}: {value}\n")

            f.write("\nFILES:\n")
            f.write("- marker1_only_4data_model.pkl: Trained model with scaler\n")
            f.write("- performance_plot.png: Performance visualization\n")
            f.write("- model_info.txt: This configuration file\n")

        print(f"Model info saved to: {info_path}")

def main():
    print("="*60)
    print("MARKER 1 DETECTION MODEL TRAINING")
    print("Training on All 4 Datasets")
    print("="*60)

    trainer = Marker1Trainer()

    data_files = [
        '../datasets/OpenBCI-RAW-2025-09-13_23-37-22.csv',
        '../datasets/OpenBCI-RAW-2025-09-13_23-45-56.csv',
        '../datasets/OpenBCI-RAW-2025-09-13_23-58-04.csv',
        '../datasets/daniel.csv'
    ]

    features_df, labels, dataset_stats = trainer.process_datasets(data_files)

    if len(features_df) == 0:
        print("Error: No data extracted from files")
        return

    X_test, y_test = trainer.train_model(features_df.values, labels)

    accuracy, roc_auc = trainer.evaluate_model(X_test, y_test)

    trainer.save_model(accuracy, roc_auc, dataset_stats)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Final Test Accuracy: {accuracy*100:.1f}%")
    print(f"Final ROC AUC: {roc_auc:.3f}")
    print("="*60)

if __name__ == "__main__":
    main()