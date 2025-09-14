# EEG Touch Detection Classifier

## Overview
This project implements a machine learning classifier to detect touch events from EEG data using symmetric channel differences and frequency band features.

## System Architecture

### Data Processing Pipeline
1. **Sample Rate**: 250 Hz (note: CSV header shows 125 Hz but treating as 250 Hz per user specification)
2. **EEG Channels**: C3, C4, P3, P4, P7, P8, T7, T8 (10-20 system)
3. **Marker System**: Touch events marked with values 1.0 and 2.0
4. **Marker Propagation**: 49 samples before, 200 samples after each marker

### Feature Extraction
- **Channel Pairs**: 4 symmetric differences (C3-C4, P3-P4, P7-P8, T7-T8)
- **Frequency Bands**: Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-50 Hz)
- **Window Size**: 250 samples (1 second)
- **Overlap**: 50% for training (125 samples)
- **Features per Window**: 160 total
  - Mean power, standard deviation, peak-to-peak, band power for each band
  - Band power ratios between frequency bands
  - Skewness and kurtosis

### Model Performance
- **Best Model**: Random Forest
- **Test Accuracy**: 69.6%
- **ROC AUC**: 0.698
- **Class Distribution**: 39.1% touch samples after marker propagation

## File Structure

```
model_training/
├── preprocess.py       # Data loading, normalization, channel mapping, marker propagation
├── features.py         # Feature extraction pipeline with windowing
├── train_model.py      # Model training with RandomForest, LogisticRegression, SVM
├── inference.py        # Real-time inference with circular buffer and voting
├── utils.py           # Visualization and analysis utilities
├── touch_detection_model.pkl  # Trained model (generated after training)
└── OpenBCI-RAW-2025-09-13_23-58-04.csv  # Raw EEG data
```

## Usage

### Training a Model
```bash
python train_model.py
```
This will:
1. Load and preprocess the EEG data
2. Extract features using sliding windows
3. Train multiple classifiers (Random Forest, Logistic Regression, SVM)
4. Save the best model to `touch_detection_model.pkl`

### Real-Time Inference
```bash
python inference.py
```
This simulates real-time processing with:
- Circular buffer for maintaining 250-sample windows
- Predictions every 25 samples (10 Hz)
- Voting window of 5 predictions for stability
- Debouncing to prevent rapid state changes

### Data Analysis
```bash
python utils.py
```
Generates visualizations and quality reports including:
- EEG channel plots with touch markers
- Frequency spectrum analysis
- Touch event distribution
- Data quality assessment

## Key Parameters

### Preprocessing
- **Normalization**: Z-score normalization per channel
- **Filtering**: 4th order Butterworth bandpass filters
- **Artifact Handling**: Replace inf with NaN, fill NaN with 0

### Inference Configuration
- **Prediction Rate**: Every 25 samples (10 Hz)
- **Voting Threshold**: 60% agreement required
- **Debounce Time**: 200ms minimum touch duration
- **Confidence Threshold**: Adjustable based on use case

## Current Limitations

1. **Moderate Accuracy**: ~70% accuracy suggests room for improvement
2. **Feature Engineering**: Current features may not fully capture somatosensory response patterns
3. **Class Imbalance**: Even after SMOTE, model shows bias toward no-touch predictions
4. **Temporal Dynamics**: May benefit from RNN/LSTM for temporal patterns

## Recommendations for Improvement

1. **Advanced Features**:
   - Event-related potentials (ERPs) around touch onset
   - Cross-frequency coupling metrics
   - Connectivity measures between regions

2. **Model Architecture**:
   - Deep learning models (CNN-LSTM) for automatic feature learning
   - Ensemble methods combining multiple time scales
   - Transfer learning from other EEG tasks

3. **Data Collection**:
   - More diverse touch patterns
   - Controlled experimental conditions
   - Multiple subjects for generalization

4. **Real-Time Optimization**:
   - GPU acceleration for feature extraction
   - Model quantization for faster inference
   - Adaptive thresholds based on baseline EEG

## Dependencies
- numpy
- pandas
- scipy
- scikit-learn
- imblearn
- joblib
- matplotlib
- seaborn

## Notes
- The system assumes continuous EEG streaming for real-time use
- Marker propagation is only used during training, not inference
- The model is subject-specific and may need retraining for different users