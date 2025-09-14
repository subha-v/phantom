import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from preprocess import EEGPreprocessor
from features import FeatureExtractor

class Marker1DetectionTrainer:
    """
    Binary classifier for detecting Marker 1 vs everything else (Marker 2 or no marker)
    Class 0: Not Marker 1 (includes Marker 0 and Marker 2)
    Class 1: Marker 1
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None

    def create_marker1_labels(self, markers: np.ndarray) -> np.ndarray:
        """
        Convert markers to binary labels:
        - Label 1: Marker value is 1.0
        - Label 0: Marker value is 0.0 or 2.0
        """
        return (markers == 1.0).astype(int)

    def prepare_data(self, features_df: pd.DataFrame, labels: np.ndarray,
                    test_size: float = 0.3, balance_method: str = 'smote') -> Tuple:
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, labels, test_size=test_size,
            random_state=self.random_state, stratify=labels
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5,
            random_state=self.random_state, stratify=y_test
        )

        print(f"Original training class distribution:")
        print(f"  Not Marker 1: {np.sum(y_train == 0)}, Marker 1: {np.sum(y_train == 1)}")

        if balance_method == 'smote' and np.sum(y_train == 1) > 1:
            try:
                smote = SMOTE(random_state=self.random_state)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                print(f"After SMOTE balancing:")
                print(f"  Not Marker 1: {np.sum(y_train_balanced == 0)}, Marker 1: {np.sum(y_train_balanced == 1)}")
            except:
                print("SMOTE failed, using original imbalanced data")
                X_train_balanced, y_train_balanced = X_train, y_train
        elif balance_method == 'undersample':
            X_train_balanced, y_train_balanced = self._undersample_majority(X_train, y_train)
            print(f"After undersampling:")
            print(f"  Not Marker 1: {np.sum(y_train_balanced == 0)}, Marker 1: {np.sum(y_train_balanced == 1)}")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train

        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        return (X_train_scaled, X_val_scaled, X_test_scaled,
                y_train_balanced, y_val, y_test)

    def _undersample_majority(self, X: pd.DataFrame, y: np.ndarray) -> Tuple:
        df = X.copy()
        df['label'] = y

        majority_class = df[df['label'] == 0]
        minority_class = df[df['label'] == 1]

        if len(minority_class) == 0:
            return X, y

        majority_downsampled = resample(majority_class,
                                      replace=False,
                                      n_samples=len(minority_class),
                                      random_state=self.random_state)

        df_balanced = pd.concat([majority_downsampled, minority_class])
        df_balanced = df_balanced.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        X_balanced = df_balanced.drop('label', axis=1)
        y_balanced = df_balanced['label'].values

        return X_balanced, y_balanced

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> RandomForestClassifier:
        print("\nTraining Random Forest...")

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)

        best_rf = grid_search.best_estimator_
        val_score = best_rf.score(X_val, y_val)
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Validation accuracy: {val_score:.3f}")

        return best_rf

    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray) -> LogisticRegression:
        print("\nTraining Logistic Regression...")

        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }

        lr = LogisticRegression(random_state=self.random_state, max_iter=1000)
        grid_search = GridSearchCV(lr, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)

        best_lr = grid_search.best_estimator_
        val_score = best_lr.score(X_val, y_val)
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Validation accuracy: {val_score:.3f}")

        return best_lr

    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray) -> SVC:
        print("\nTraining SVM...")

        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }

        svm = SVC(random_state=self.random_state, probability=True)
        grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)

        best_svm = grid_search.best_estimator_
        val_score = best_svm.score(X_val, y_val)
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Validation accuracy: {val_score:.3f}")

        return best_svm

    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str) -> Dict[str, Any]:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        accuracy = model.score(X_test, y_test)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        if y_pred_proba is not None:
            results['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            results['fpr'], results['tpr'], _ = roc_curve(y_test, y_pred_proba)

        return results

    def plot_results(self, results: Dict[str, Any], feature_importance: np.ndarray = None,
                    feature_names: list = None, save_path: str = 'marker1_performance.png'):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', ax=axes[0, 0],
                   xticklabels=['Not Marker 1', 'Marker 1'],
                   yticklabels=['Not Marker 1', 'Marker 1'])
        axes[0, 0].set_title(f"Confusion Matrix - {results['model_name']}")
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')

        if 'fpr' in results and 'tpr' in results:
            axes[0, 1].plot(results['fpr'], results['tpr'],
                          label=f"ROC (AUC = {results['roc_auc']:.3f})")
            axes[0, 1].plot([0, 1], [0, 1], 'k--')
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve')
            axes[0, 1].legend()

        report_df = pd.DataFrame(results['classification_report']).T
        report_text = f"Accuracy: {results['accuracy']:.3f}\n\n"
        report_text += f"Precision (Marker 1): {report_df.loc['1', 'precision']:.3f}\n"
        report_text += f"Recall (Marker 1): {report_df.loc['1', 'recall']:.3f}\n"
        report_text += f"F1-Score (Marker 1): {report_df.loc['1', 'f1-score']:.3f}"
        axes[1, 0].text(0.1, 0.5, report_text, fontsize=12, verticalalignment='center')
        axes[1, 0].set_title('Performance Metrics')
        axes[1, 0].axis('off')

        if feature_importance is not None and feature_names is not None:
            top_n = 15
            indices = np.argsort(feature_importance)[-top_n:]
            top_features = [feature_names[i] for i in indices]
            top_importance = feature_importance[indices]

            axes[1, 1].barh(range(len(top_features)), top_importance)
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features, fontsize=8)
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title(f'Top {top_n} Features')

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()

    def train_all_models(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                        y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                        feature_names: list) -> Dict[str, Any]:
        self.models['RandomForest'] = self.train_random_forest(X_train, y_train, X_val, y_val)
        self.models['LogisticRegression'] = self.train_logistic_regression(X_train, y_train, X_val, y_val)
        self.models['SVM'] = self.train_svm(X_train, y_train, X_val, y_val)

        print("\n" + "="*50)
        print("Evaluating models on test set...")
        best_score = 0
        all_results = {}

        for name, model in self.models.items():
            results = self.evaluate_model(model, X_test, y_test, name)
            all_results[name] = results

            print(f"\n{name}:")
            print(f"  Test Accuracy: {results['accuracy']:.3f}")
            if 'roc_auc' in results:
                print(f"  ROC AUC: {results['roc_auc']:.3f}")

            if results['accuracy'] > best_score:
                best_score = results['accuracy']
                self.best_model = model
                self.best_model_name = name

                if hasattr(model, 'feature_importances_'):
                    self.feature_importance = model.feature_importances_

        print(f"\n{'='*50}")
        print(f"Best Model: {self.best_model_name} (Accuracy: {best_score:.3f})")

        if self.feature_importance is not None:
            self.plot_results(all_results[self.best_model_name],
                            self.feature_importance, feature_names,
                            'marker1_performance.png')

        return all_results

    def save_model(self, filepath: str = 'marker1_detection_model.pkl'):
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'model_name': self.best_model_name,
            'feature_importance': self.feature_importance,
            'task': 'marker1_detection'
        }
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to {filepath}")

def extract_features_for_marker1(preprocessed_df: pd.DataFrame, overlap: float = 0.5):
    """
    Custom feature extraction that creates labels for Marker 1 detection
    """
    extractor = FeatureExtractor(sample_rate=250, window_size=250)

    all_features = []
    all_labels = []

    markers = preprocessed_df['Marker'].values if 'Marker' in preprocessed_df.columns else None

    channel_pairs = ['C3-C4', 'P3-P4', 'P7-P8', 'T7-T8']

    for pair in channel_pairs:
        if pair not in preprocessed_df.columns:
            print(f"Warning: Channel pair {pair} not found in dataframe")
            continue

        signal_data = preprocessed_df[pair].values
        windows, _ = extractor.create_sliding_windows(signal_data, None, overlap)

        print(f"Processing {len(windows)} windows for {pair}...")

        step_size = int(250 * (1 - overlap))

        for i, window in enumerate(windows):
            window_features = extractor.extract_all_features_for_window(window)

            for key in list(window_features.keys()):
                window_features[f'{pair}_{key}'] = window_features.pop(key)

            all_features.append(window_features)

            if markers is not None:
                start = i * step_size
                end = start + 250
                window_markers = markers[start:end]
                # Label is 1 if ANY marker in window is 1.0, else 0
                label = 1 if np.any(window_markers == 1.0) else 0
                all_labels.append(label)

    features_df = pd.DataFrame(all_features)
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.fillna(0)

    labels_array = np.array(all_labels) if all_labels else None

    return features_df, labels_array

def main():
    print("="*50)
    print("EEG Marker 1 Detection Model Training")
    print("Binary Classification: Marker 1 vs Not Marker 1")
    print("="*50)

    preprocessor = EEGPreprocessor(sample_rate=250)
    print("\nStep 1: Loading and preprocessing data...")
    _, df_differences = preprocessor.preprocess_pipeline(
        "OpenBCI-RAW-2025-09-13_23-58-04.csv"
    )

    print("\nStep 2: Extracting features with Marker 1 specific labels...")
    features_df, labels = extract_features_for_marker1(df_differences, overlap=0.5)

    print(f"Feature extraction complete:")
    print(f"  Total features: {features_df.shape[1]}")
    print(f"  Total samples: {features_df.shape[0]}")
    if labels is not None:
        print(f"  Marker 1 samples: {np.sum(labels == 1)}")
        print(f"  Not Marker 1 samples: {np.sum(labels == 0)}")

        # Analyze the "Not Marker 1" category
        df_normalized = preprocessor.load_data("OpenBCI-RAW-2025-09-13_23-58-04.csv")
        markers = df_normalized['Marker'].values
        marker_2_count = np.sum(markers == 2.0)
        marker_0_count = np.sum(markers == 0.0)
        marker_1_count = np.sum(markers == 1.0)
        print(f"\nOriginal marker distribution in data:")
        print(f"  Marker 0 (no touch): {marker_0_count}")
        print(f"  Marker 1: {marker_1_count}")
        print(f"  Marker 2: {marker_2_count}")

    if labels is None or len(np.unique(labels)) < 2:
        print("Error: Insufficient data or labels for training")
        return

    trainer = Marker1DetectionTrainer(random_state=42)
    print("\nStep 3: Preparing train/validation/test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        features_df, labels, test_size=0.3, balance_method='smote'
    )

    print("\nStep 4: Training models...")
    results = trainer.train_all_models(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        features_df.columns.tolist()
    )

    print("\nStep 5: Saving best model...")
    trainer.save_model('marker1_detection_model.pkl')

    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)

if __name__ == "__main__":
    main()