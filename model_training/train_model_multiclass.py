import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                           precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

from preprocess import EEGPreprocessor
from features import FeatureExtractor

class RubberHandIllusionClassifier:
    """
    Multiclass classifier for rubber hand illusion EEG detection

    Classes:
    - Class 0: No stimulation (marker = 0)
    - Class 1: Both hands touched (marker = 1) - real + fake hand
    - Class 2: Fake hand only touched (marker = 2)

    This classifier is designed to detect the neural correlates of the rubber hand
    illusion, where synchronous touching of real and fake hands can create a sense
    of ownership over the fake hand.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        self.class_names = ['No Touch', 'Both Hands', 'Fake Hand Only']
        self.n_classes = 3

    def extract_multiclass_features(self, preprocessed_df: pd.DataFrame,
                                   overlap: float = 0.5) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Extract features with multiclass labels (0, 1, 2)
        """
        extractor = FeatureExtractor(sample_rate=250, window_size=250)

        all_features = []
        all_labels = []

        markers = preprocessed_df['Marker'].values if 'Marker' in preprocessed_df.columns else None

        if markers is None:
            raise ValueError("No Marker column found in data")

        channel_pairs = ['C3-C4', 'P3-P4', 'P7-P8', 'T7-T8']

        for pair in channel_pairs:
            if pair not in preprocessed_df.columns:
                print(f"Warning: Channel pair {pair} not found")
                continue

            signal_data = preprocessed_df[pair].values
            windows, _ = extractor.create_sliding_windows(signal_data, None, overlap)

            print(f"Processing {len(windows)} windows for {pair}...")

            step_size = int(250 * (1 - overlap))

            for i, window in enumerate(windows):
                window_features = extractor.extract_all_features_for_window(window)

                # Prefix features with channel pair name
                for key in list(window_features.keys()):
                    window_features[f'{pair}_{key}'] = window_features.pop(key)

                all_features.append(window_features)

                # Extract multiclass label
                start = i * step_size
                end = start + 250
                window_markers = markers[start:end]

                # Determine the dominant marker in the window
                if np.any(window_markers == 1.0):
                    label = 1  # Both hands touched
                elif np.any(window_markers == 2.0):
                    label = 2  # Fake hand only
                else:
                    label = 0  # No touch

                all_labels.append(label)

        features_df = pd.DataFrame(all_features)
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)

        labels_array = np.array(all_labels)

        return features_df, labels_array

    def prepare_data(self, features_df: pd.DataFrame, labels: np.ndarray,
                    test_size: float = 0.3, balance_method: str = 'class_weight') -> Tuple:
        """
        Prepare data with proper stratification for multiclass
        """
        # Stratified split to maintain class proportions
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, labels, test_size=test_size,
            random_state=self.random_state, stratify=labels
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5,
            random_state=self.random_state, stratify=y_test
        )

        print(f"Original training class distribution:")
        for i in range(self.n_classes):
            print(f"  {self.class_names[i]}: {np.sum(y_train == i)}")

        # Handle class imbalance
        if balance_method == 'smote':
            try:
                smote = SMOTE(random_state=self.random_state, k_neighbors=min(5, min(np.bincount(y_train))-1))
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                print(f"\nAfter SMOTE balancing:")
                for i in range(self.n_classes):
                    print(f"  {self.class_names[i]}: {np.sum(y_train_balanced == i)}")
            except:
                print("SMOTE failed, using original data")
                X_train_balanced, y_train_balanced = X_train, y_train
        else:
            X_train_balanced, y_train_balanced = X_train, y_train

        # Compute class weights for models that support it
        self.class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train_balanced),
            y=y_train_balanced
        )
        self.class_weight_dict = {i: w for i, w in enumerate(self.class_weights)}

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        return (X_train_scaled, X_val_scaled, X_test_scaled,
                y_train_balanced, y_val, y_test)

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> RandomForestClassifier:
        """Train Random Forest with multiclass support"""
        print("\nTraining Random Forest (Multiclass)...")

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced', self.class_weight_dict]
        }

        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)

        # Use stratified k-fold for cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)

        grid_search = GridSearchCV(rf, param_grid, cv=cv,
                                 scoring='f1_macro', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)

        best_rf = grid_search.best_estimator_
        val_score = best_rf.score(X_val, y_val)
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Validation accuracy: {val_score:.3f}")

        return best_rf

    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray) -> LogisticRegression:
        """Train Logistic Regression with multiclass support"""
        print("\nTraining Logistic Regression (Multiclass)...")

        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs'],
            'multi_class': ['multinomial'],
            'class_weight': ['balanced', self.class_weight_dict]
        }

        lr = LogisticRegression(random_state=self.random_state, max_iter=1000)

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)

        grid_search = GridSearchCV(lr, param_grid, cv=cv,
                                 scoring='f1_macro', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)

        best_lr = grid_search.best_estimator_
        val_score = best_lr.score(X_val, y_val)
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Validation accuracy: {val_score:.3f}")

        return best_lr

    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray) -> SVC:
        """Train SVM with multiclass support (one-vs-rest)"""
        print("\nTraining SVM (Multiclass)...")

        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto'],
            'decision_function_shape': ['ovr'],  # one-vs-rest for multiclass
            'class_weight': ['balanced', self.class_weight_dict]
        }

        svm = SVC(random_state=self.random_state, probability=True)

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)

        grid_search = GridSearchCV(svm, param_grid, cv=cv,
                                 scoring='f1_macro', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)

        best_svm = grid_search.best_estimator_
        val_score = best_svm.score(X_val, y_val)
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Validation accuracy: {val_score:.3f}")

        return best_svm

    def evaluate_multiclass_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                                 model_name: str) -> Dict[str, Any]:
        """Comprehensive evaluation for multiclass model"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        # Overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, labels=[0, 1, 2]
        )

        # Macro and weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro'
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )

        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'per_class_precision': precision,
            'per_class_recall': recall,
            'per_class_f1': f1,
            'per_class_support': support,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred,
                                                          target_names=self.class_names)
        }

        # Multiclass ROC AUC (one-vs-rest)
        if y_pred_proba is not None:
            try:
                results['roc_auc_ovr'] = roc_auc_score(y_test, y_pred_proba,
                                                      multi_class='ovr', average='macro')
                results['roc_auc_ovo'] = roc_auc_score(y_test, y_pred_proba,
                                                      multi_class='ovo', average='macro')
            except:
                results['roc_auc_ovr'] = None
                results['roc_auc_ovo'] = None

        return results

    def plot_multiclass_results(self, results: Dict[str, Any],
                               feature_importance: np.ndarray = None,
                               feature_names: list = None,
                               save_path: str = 'multiclass_performance.png'):
        """Create comprehensive visualization for multiclass results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Confusion Matrix
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d',
                   ax=axes[0, 0], cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        axes[0, 0].set_title(f"Confusion Matrix - {results['model_name']}")
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')

        # Normalized Confusion Matrix
        cm_normalized = results['confusion_matrix'].astype('float') / results['confusion_matrix'].sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f',
                   ax=axes[0, 1], cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        axes[0, 1].set_title('Normalized Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')

        # Per-class metrics
        metrics_df = pd.DataFrame({
            'Precision': results['per_class_precision'],
            'Recall': results['per_class_recall'],
            'F1-Score': results['per_class_f1']
        }, index=self.class_names)

        metrics_df.plot(kind='bar', ax=axes[0, 2])
        axes[0, 2].set_title('Per-Class Performance')
        axes[0, 2].set_xlabel('Class')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].legend(loc='lower right')
        axes[0, 2].set_ylim([0, 1])
        axes[0, 2].grid(axis='y', alpha=0.3)

        # Overall metrics text
        metrics_text = f"Overall Accuracy: {results['accuracy']:.3f}\n\n"
        metrics_text += f"Macro F1-Score: {results['macro_f1']:.3f}\n"
        metrics_text += f"Weighted F1-Score: {results['weighted_f1']:.3f}\n"
        if results.get('roc_auc_ovr'):
            metrics_text += f"\nROC AUC (One-vs-Rest): {results['roc_auc_ovr']:.3f}\n"
            metrics_text += f"ROC AUC (One-vs-One): {results['roc_auc_ovo']:.3f}"

        axes[1, 0].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
        axes[1, 0].set_title('Overall Performance')
        axes[1, 0].axis('off')

        # Class distribution in predictions
        unique, counts = np.unique(results['predictions'], return_counts=True)
        axes[1, 1].bar([self.class_names[i] for i in unique], counts)
        axes[1, 1].set_title('Prediction Distribution')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Count')

        # Feature importance (if available)
        if feature_importance is not None and feature_names is not None:
            top_n = 15
            indices = np.argsort(feature_importance)[-top_n:]
            top_features = [feature_names[i] for i in indices]
            top_importance = feature_importance[indices]

            axes[1, 2].barh(range(len(top_features)), top_importance)
            axes[1, 2].set_yticks(range(len(top_features)))
            axes[1, 2].set_yticklabels(top_features, fontsize=8)
            axes[1, 2].set_xlabel('Importance')
            axes[1, 2].set_title(f'Top {top_n} Features')
        else:
            axes[1, 2].axis('off')

        plt.suptitle('Multiclass Rubber Hand Illusion Detection Performance', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()

        return fig

    def train_all_models(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                        y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                        feature_names: list) -> Dict[str, Any]:
        """Train and evaluate all multiclass models"""

        self.models['RandomForest'] = self.train_random_forest(X_train, y_train, X_val, y_val)
        self.models['LogisticRegression'] = self.train_logistic_regression(X_train, y_train, X_val, y_val)
        self.models['SVM'] = self.train_svm(X_train, y_train, X_val, y_val)

        print("\n" + "="*50)
        print("Evaluating models on test set...")
        best_score = 0
        all_results = {}

        for name, model in self.models.items():
            results = self.evaluate_multiclass_model(model, X_test, y_test, name)
            all_results[name] = results

            print(f"\n{name}:")
            print(f"  Test Accuracy: {results['accuracy']:.3f}")
            print(f"  Macro F1-Score: {results['macro_f1']:.3f}")
            print(f"  Per-class F1: {[f'{c}: {f:.3f}' for c, f in zip(self.class_names, results['per_class_f1'])]}")

            # Use macro F1 as the selection metric for multiclass
            if results['macro_f1'] > best_score:
                best_score = results['macro_f1']
                self.best_model = model
                self.best_model_name = name

                if hasattr(model, 'feature_importances_'):
                    self.feature_importance = model.feature_importances_

        print(f"\n{'='*50}")
        print(f"Best Model: {self.best_model_name} (Macro F1: {best_score:.3f})")

        # Plot results for best model
        self.plot_multiclass_results(all_results[self.best_model_name],
                                    self.feature_importance, feature_names)

        # Print detailed classification report
        print("\nDetailed Classification Report (Best Model):")
        print(all_results[self.best_model_name]['classification_report'])

        return all_results

    def save_model(self, filepath: str = 'multiclass_model.pkl'):
        """Save the trained multiclass model"""
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'model_name': self.best_model_name,
            'feature_importance': self.feature_importance,
            'class_names': self.class_names,
            'n_classes': self.n_classes,
            'task': 'multiclass_rubber_hand_illusion'
        }
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to {filepath}")


def main():
    print("="*60)
    print("EEG Rubber Hand Illusion Multiclass Detection")
    print("Classes: No Touch | Both Hands | Fake Hand Only")
    print("="*60)

    # Initialize preprocessor and load data
    preprocessor = EEGPreprocessor(sample_rate=250)
    print("\nStep 1: Loading and preprocessing data...")
    _, df_differences = preprocessor.preprocess_pipeline(
        "OpenBCI-RAW-2025-09-13_23-58-04.csv"
    )

    # Initialize classifier
    classifier = RubberHandIllusionClassifier(random_state=42)

    # Extract features with multiclass labels
    print("\nStep 2: Extracting features with multiclass labels...")
    features_df, labels = classifier.extract_multiclass_features(df_differences, overlap=0.5)

    print(f"\nFeature extraction complete:")
    print(f"  Total features: {features_df.shape[1]}")
    print(f"  Total samples: {features_df.shape[0]}")

    # Print class distribution
    print(f"\nClass distribution:")
    for i in range(classifier.n_classes):
        count = np.sum(labels == i)
        percentage = 100 * count / len(labels)
        print(f"  {classifier.class_names[i]} (class {i}): {count} ({percentage:.1f}%)")

    if len(np.unique(labels)) < classifier.n_classes:
        print("\nWarning: Not all classes present in data!")

    # Prepare data
    print("\nStep 3: Preparing train/validation/test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = classifier.prepare_data(
        features_df, labels, test_size=0.3, balance_method='class_weight'
    )

    # Train models
    print("\nStep 4: Training multiclass models...")
    results = classifier.train_all_models(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        features_df.columns.tolist()
    )

    # Save model
    print("\nStep 5: Saving best multiclass model...")
    classifier.save_model('multiclass_model.pkl')

    print("\n" + "="*60)
    print("Multiclass training complete!")
    print("="*60)


if __name__ == "__main__":
    main()