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
from typing import Tuple
import warnings
import os
import sys
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from preprocess import EEGPreprocessor
from features import FeatureExtractor

class NewModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        # Use 500Hz sampling rate for new 8-channel setup
        self.preprocessor = EEGPreprocessor(sample_rate=500)
        self.extractor = FeatureExtractor(sample_rate=500, window_size=500)

    def propagate_markers(self, markers: np.ndarray, before_samples: int = 98, after_samples: int = 400) -> np.ndarray:
        """Propagate markers with windows adjusted for 500Hz sampling"""
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

    def train_model(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.3):
        """Train binary classification model with extreme class imbalance handling"""
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
            # Use higher sampling strategy due to extreme imbalance
            smote = SMOTE(random_state=42, sampling_strategy=0.1)  # Create synthetic samples to get 10% minority class
            try:
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                print(f"After SMOTE:")
                print(f"  Touch: {np.sum(y_train_balanced == 1)}")
                print(f"  No touch: {np.sum(y_train_balanced == 0)}")
            except:
                print("SMOTE failed due to extreme imbalance, using class weights instead")
                X_train_balanced, y_train_balanced = X_train, y_train
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
            print("Skipping SMOTE (insufficient samples)")

        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"\n{'='*50}")
        print("TRAINING RANDOM FOREST MODEL")
        print('='*50)

        # Adjusted hyperparameters for extreme imbalance
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [5, 10, 15],  # Reduced depth to prevent overfitting
            'min_samples_split': [5, 10],  # Increased to prevent overfitting
            'min_samples_leaf': [2, 4],  # Increased to prevent overfitting
            'class_weight': ['balanced', 'balanced_subsample']  # Handle imbalance
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
        """Evaluate model performance on test set"""
        print(f"\n{'='*50}")
        print("MODEL EVALUATION")
        print('='*50)

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        
        # Check if we have both classes in test set for ROC AUC
        if len(np.unique(y_test)) > 1:
            roc_auc = roc_auc_score(y_test, y_proba)
        else:
            roc_auc = 0.0
            print("Warning: Only one class in test set, ROC AUC cannot be computed")

        print(f"Test Accuracy: {accuracy:.3f}")
        print(f"ROC AUC: {roc_auc:.3f}")

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

        self.plot_results(y_test, y_pred, y_proba, accuracy, roc_auc)

        return accuracy, roc_auc

    def plot_results(self, y_test, y_pred, y_proba, accuracy, roc_auc):
        """Generate performance visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Touch', 'Touch'],
                   yticklabels=['No Touch', 'Touch'], ax=axes[0, 0])
        axes[0, 0].set_title(f'Confusion Matrix (Acc: {accuracy:.3f})')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')

        # ROC Curve
        if roc_auc > 0:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            axes[0, 1].plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
            axes[0, 1].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        else:
            axes[0, 1].text(0.5, 0.5, 'ROC curve not available\n(single class in test set)',
                          ha='center', va='center')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True, alpha=0.3)

        # Feature Importance
        feature_importance = self.model.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-20:]
        top_features = feature_importance[top_features_idx]

        axes[1, 0].barh(range(len(top_features)), top_features, color='green', alpha=0.7)
        axes[1, 0].set_xlabel('Feature Importance')
        axes[1, 0].set_title('Top 20 Most Important Features')
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels([f'Feature {i}' for i in top_features_idx])

        # Calibration Plot
        if len(np.unique(y_test)) > 1:
            from sklearn.calibration import calibration_curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, y_proba, n_bins=10
            )
            axes[1, 1].plot(mean_predicted_value, fraction_of_positives, 's-', color='blue', label='Model')
            axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        else:
            axes[1, 1].text(0.5, 0.5, 'Calibration plot not available\n(single class in test set)',
                          ha='center', va='center')
        axes[1, 1].set_xlabel('Mean predicted probability')
        axes[1, 1].set_ylabel('Fraction of positives')
        axes[1, 1].set_title('Calibration Plot')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        
        # Create model directory if it doesn't exist
        os.makedirs('../models/new_model', exist_ok=True)
        plt.savefig('../models/new_model/performance_plot.png', dpi=100, bbox_inches='tight')
        plt.show()
        print("\nPerformance plot saved as 'performance_plot.png'")

    def save_model(self, accuracy: float, roc_auc: float, dataset_path: str):
        """Save trained model and metadata"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'sample_rate': 500,
            'window_size': 500
        }

        # Create model directory if it doesn't exist
        os.makedirs('../models/new_model', exist_ok=True)
        
        model_path = '../models/new_model/new_model.pkl'
        joblib.dump(model_data, model_path)
        print(f"\nModel saved to: {model_path}")

        info_path = '../models/new_model/model_info.txt'
        with open(info_path, 'w') as f:
            f.write("MODEL: new_model\n")
            f.write("="*50 + "\n")
            f.write("Binary Classification Model for Touch Detection\n\n")
            f.write(f"Training Date: 2025-09-14\n")
            f.write(f"Model Type: Random Forest Classifier\n")
            f.write(f"Task: Binary classification (Touch vs No Touch)\n\n")

            f.write("CLASS MAPPING:\n")
            f.write("- Class 0: No Touch (marker = 0)\n")
            f.write("- Class 1: Touch (marker = 1)\n\n")

            f.write("PERFORMANCE METRICS:\n")
            f.write(f"- Test Accuracy: {accuracy*100:.1f}%\n")
            f.write(f"- ROC AUC: {roc_auc:.3f}\n\n")

            f.write(f"TRAINING DATA:\n")
            f.write(f"- Dataset: {os.path.basename(dataset_path)}\n")
            f.write(f"- Note: Severe class imbalance (41 positive samples)\n\n")

            f.write("PREPROCESSING:\n")
            f.write("- Sample Rate: 500 Hz\n")
            f.write("- Window Size: 500 samples (1 second)\n")
            f.write("- Overlap: 50%\n")
            f.write("- Marker Propagation: 98 samples before, 400 samples after\n")
            f.write("- Class Balancing: SMOTE (if possible) or class weights\n\n")

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
            f.write("- new_model.pkl: Trained model with scaler\n")
            f.write("- performance_plot.png: Performance visualization\n")
            f.write("- model_info.txt: This configuration file\n")

        print(f"Model info saved to: {info_path}")

def main():
    print("="*60)
    print("NEW MODEL TRAINING - BINARY CLASSIFICATION")
    print("Training on daniel_updated.csv (500Hz, 8 channels)")
    print("="*60)

    trainer = NewModelTrainer()

    data_file = '../datasets/daniel_updated.csv'
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"Error: Dataset not found at {data_file}")
        return

    features_df, labels = trainer.process_dataset(data_file)

    if len(features_df) == 0:
        print("Error: No data extracted from file")
        return

    print(f"\n{'='*50}")
    print("DATASET SUMMARY")
    print('='*50)
    print(f"Total windows: {len(features_df)}")
    print(f"Feature dimensions: {features_df.shape}")
    print(f"Touch windows: {np.sum(labels == 1)}")
    print(f"No touch windows: {np.sum(labels == 0)}")
    
    # Warning about extreme imbalance
    if np.sum(labels == 1) < 50:
        print("\n⚠️  WARNING: Extremely low number of positive samples!")
        print("Model may have difficulty learning touch patterns.")
        print("Consider collecting more touch data for better performance.")

    X_test, y_test = trainer.train_model(features_df.values, labels)

    accuracy, roc_auc = trainer.evaluate_model(X_test, y_test)

    trainer.save_model(accuracy, roc_auc, data_file)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Final Test Accuracy: {accuracy*100:.1f}%")
    print(f"Final ROC AUC: {roc_auc:.3f}")
    print("="*60)

if __name__ == "__main__":
    main()