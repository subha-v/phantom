import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from preprocess import EEGPreprocessor
from features import FeatureExtractor

class ModelFineTuner:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.original_model = None
        self.scaler = None
        self.new_model = None

    def load_pretrained_model(self):
        print("Loading pretrained model...")
        model_data = joblib.load(self.model_path)

        if isinstance(model_data, dict):
            self.original_model = model_data.get('model')
            self.scaler = model_data.get('scaler')
        else:
            self.original_model = model_data
            self.scaler = StandardScaler()

        print(f"Model type: {type(self.original_model).__name__}")
        print(f"Number of estimators: {self.original_model.n_estimators}")
        return self.original_model, self.scaler

    def preprocess_new_data(self, file_paths: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
        preprocessor = EEGPreprocessor(sample_rate=125)
        extractor = FeatureExtractor(sample_rate=125, window_size=125)

        all_features = []
        all_labels = []

        for file_path in file_paths:
            print(f"\nProcessing {os.path.basename(file_path)}...")

            _, df_differences = preprocessor.preprocess_pipeline(file_path)

            if 'Marker' in df_differences.columns:
                markers = df_differences['Marker'].values
                propagated_markers = self.propagate_markers(markers, before_samples=24, after_samples=100)
                df_differences['Marker'] = propagated_markers

            features_df, labels = extractor.extract_features_pipeline(
                df_differences,
                overlap=0.5
            )

            if labels is not None:
                binary_labels = np.where((labels == 1) | (labels == 2), 1, 0)
                all_features.append(features_df)
                all_labels.append(binary_labels)

                print(f"  Extracted {len(features_df)} windows")
                print(f"  Touch windows: {np.sum(binary_labels == 1)}")
                print(f"  No-touch windows: {np.sum(binary_labels == 0)}")

        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            combined_labels = np.concatenate(all_labels)
            return combined_features, combined_labels
        else:
            return pd.DataFrame(), np.array([])

    def propagate_markers(self, markers: np.ndarray, before_samples: int, after_samples: int) -> np.ndarray:
        propagated = markers.copy()
        marker_indices = np.where(markers > 0)[0]

        for idx in marker_indices:
            start = max(0, idx - before_samples)
            end = min(len(propagated), idx + after_samples + 1)
            propagated[start:end] = markers[idx]

        return propagated

    def fine_tune_model(self, X_new: np.ndarray, y_new: np.ndarray,
                       test_size: float = 0.3) -> RandomForestClassifier:
        X_train, X_test, y_train, y_test = train_test_split(
            X_new, y_new, test_size=test_size, random_state=42, stratify=y_new
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
        )

        print(f"\nTraining set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")

        print(f"\nClass distribution in training:")
        print(f"  No-touch: {np.sum(y_train == 0)}")
        print(f"  Touch: {np.sum(y_train == 1)}")

        if np.sum(y_train == 1) > 1:
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"\nAfter SMOTE balancing:")
            print(f"  No-touch: {np.sum(y_train_balanced == 0)}")
            print(f"  Touch: {np.sum(y_train_balanced == 1)}")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train

        print("\nFine-tuning Random Forest model...")
        self.new_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
            warm_start=True
        )

        self.new_model.n_estimators = 100
        self.new_model.fit(X_train_balanced, y_train_balanced)

        for n_trees in [200, 300]:
            self.new_model.n_estimators = n_trees
            self.new_model.fit(X_train_balanced, y_train_balanced)

            val_score = self.new_model.score(X_val, y_val)
            print(f"  {n_trees} trees - Validation accuracy: {val_score:.3f}")

        return X_test, y_test

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray):
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)

        print("\nOriginal Model Performance:")
        y_pred_original = self.original_model.predict(X_test)
        y_proba_original = self.original_model.predict_proba(X_test)[:, 1]

        acc_original = accuracy_score(y_test, y_pred_original)
        roc_original = roc_auc_score(y_test, y_proba_original)

        print(f"  Accuracy: {acc_original:.3f}")
        print(f"  ROC AUC: {roc_original:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_original,
                                   target_names=['No-touch', 'Touch']))

        print("\nFine-tuned Model Performance:")
        y_pred_new = self.new_model.predict(X_test)
        y_proba_new = self.new_model.predict_proba(X_test)[:, 1]

        acc_new = accuracy_score(y_test, y_pred_new)
        roc_new = roc_auc_score(y_test, y_proba_new)

        print(f"  Accuracy: {acc_new:.3f}")
        print(f"  ROC AUC: {roc_new:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_new,
                                   target_names=['No-touch', 'Touch']))

        print(f"\nImprovement:")
        print(f"  Accuracy: {(acc_new - acc_original):.3f} ({(acc_new - acc_original)*100:.1f}%)")
        print(f"  ROC AUC: {(roc_new - roc_original):.3f}")

        self.plot_comparison(y_test, y_pred_original, y_pred_new,
                           acc_original, acc_new, roc_original, roc_new)

        return acc_new, roc_new

    def plot_comparison(self, y_test, y_pred_original, y_pred_new,
                       acc_orig, acc_new, roc_orig, roc_new):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        cm_original = confusion_matrix(y_test, y_pred_original)
        sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No-touch', 'Touch'],
                   yticklabels=['No-touch', 'Touch'], ax=axes[0, 0])
        axes[0, 0].set_title(f'Original Model (Acc: {acc_orig:.3f})')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')

        cm_new = confusion_matrix(y_test, y_pred_new)
        sns.heatmap(cm_new, annot=True, fmt='d', cmap='Greens',
                   xticklabels=['No-touch', 'Touch'],
                   yticklabels=['No-touch', 'Touch'], ax=axes[0, 1])
        axes[0, 1].set_title(f'Fine-tuned Model (Acc: {acc_new:.3f})')
        axes[0, 1].set_ylabel('True Label')
        axes[0, 1].set_xlabel('Predicted Label')

        metrics = ['Accuracy', 'ROC AUC']
        original_scores = [acc_orig, roc_orig]
        new_scores = [acc_new, roc_new]

        x = np.arange(len(metrics))
        width = 0.35

        axes[1, 0].bar(x - width/2, original_scores, width, label='Original', color='blue', alpha=0.7)
        axes[1, 0].bar(x + width/2, new_scores, width, label='Fine-tuned', color='green', alpha=0.7)
        axes[1, 0].set_xlabel('Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Model Performance Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(metrics)
        axes[1, 0].legend()
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(True, alpha=0.3)

        feature_importance = self.new_model.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-20:]
        top_features = feature_importance[top_features_idx]

        axes[1, 1].barh(range(len(top_features)), top_features, color='green', alpha=0.7)
        axes[1, 1].set_xlabel('Feature Importance')
        axes[1, 1].set_title('Top 20 Most Important Features (Fine-tuned Model)')
        axes[1, 1].set_yticks(range(len(top_features)))
        axes[1, 1].set_yticklabels([f'Feature {i}' for i in top_features_idx])

        plt.tight_layout()
        plt.savefig('model_training/marker12_or_none/model_comparison.png', dpi=100, bbox_inches='tight')
        plt.show()
        print("\nComparison plot saved as 'model_comparison.png'")

    def save_finetuned_model(self, accuracy: float, roc_auc: float):
        model_data = {
            'model': self.new_model,
            'scaler': self.scaler,
            'accuracy': accuracy,
            'roc_auc': roc_auc
        }

        output_path = 'model_training/marker12_or_none/touch_detection_model_finetuned.pkl'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(model_data, output_path)
        print(f"\nFine-tuned model saved to: {output_path}")

        info_path = 'model_training/marker12_or_none/model_info_finetuned.txt'
        with open(info_path, 'w') as f:
            f.write("MODEL: marker12_or_none (FINE-TUNED)\n")
            f.write("="*40 + "\n")
            f.write("Binary Classification Model for EEG Touch Detection\n\n")
            f.write(f"Fine-tuning Date: 2025-09-14\n")
            f.write(f"Model Type: Random Forest Classifier (Fine-tuned)\n")
            f.write(f"Task: Binary classification (touch vs no-touch)\n\n")
            f.write("CLASS MAPPING:\n")
            f.write("- Class 0: No touch (marker = 0)\n")
            f.write("- Class 1: Touch (marker = 1 OR marker = 2)\n\n")
            f.write("PERFORMANCE METRICS (After Fine-tuning):\n")
            f.write(f"- Test Accuracy: {accuracy*100:.1f}%\n")
            f.write(f"- ROC AUC: {roc_auc:.3f}\n\n")
            f.write("TRAINING DATA:\n")
            f.write("- Original model: OpenBCI-RAW-2025-09-13_23-58-04.csv\n")
            f.write("- Fine-tuning data 1: OpenBCI-RAW-2025-09-13_23-37-22.csv\n")
            f.write("- Fine-tuning data 2: OpenBCI-RAW-2025-09-13_23-45-56.csv\n\n")
            f.write("FINE-TUNING CONFIGURATION:\n")
            f.write("- Method: Warm-start incremental learning\n")
            f.write("- Total estimators: 300\n")
            f.write("- Class balancing: SMOTE\n\n")
            f.write("FILES:\n")
            f.write("- touch_detection_model_finetuned.pkl: Fine-tuned model with scaler\n")
            f.write("- model_comparison.png: Performance comparison visualization\n")
            f.write("- model_info_finetuned.txt: This configuration file\n")

        print(f"Model info saved to: {info_path}")

def main():
    print("="*60)
    print("MARKER12_OR_NONE MODEL FINE-TUNING")
    print("="*60)

    model_path = '../models/marker12_or_none/touch_detection_model.pkl'
    finetuner = ModelFineTuner(model_path)

    finetuner.load_pretrained_model()

    new_data_files = [
        '../datasets/OpenBCI-RAW-2025-09-13_23-37-22.csv',
        '../datasets/OpenBCI-RAW-2025-09-13_23-45-56.csv'
    ]

    features_df, labels = finetuner.preprocess_new_data(new_data_files)

    if len(features_df) == 0:
        print("Error: No data extracted from files")
        return

    print(f"\nTotal extracted features: {len(features_df)}")
    print(f"Feature dimensions: {features_df.shape}")

    X_scaled = finetuner.scaler.fit_transform(features_df)

    X_test, y_test = finetuner.fine_tune_model(X_scaled, labels)

    accuracy, roc_auc = finetuner.evaluate_models(X_test, y_test)

    finetuner.save_finetuned_model(accuracy, roc_auc)

    print("\n" + "="*60)
    print("FINE-TUNING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()