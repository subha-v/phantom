# Model Training Directory Structure

## Overview
This directory contains EEG model training infrastructure for touch detection using OpenBCI data.

## Directory Structure

```
model_training/
├── datasets/               # All training/test datasets
│   ├── OpenBCI-RAW-2025-09-13_23-37-22.csv
│   ├── OpenBCI-RAW-2025-09-13_23-45-56.csv
│   ├── OpenBCI-RAW-2025-09-13_23-58-04.csv
│   └── daniel.csv
│
├── models/                 # Trained models organized by type
│   ├── marker1_or_not/     # Binary: Marker 1 detection only
│   │   ├── marker1_detection_model.pkl
│   │   ├── marker1_performance.png
│   │   └── model_info.txt
│   │
│   ├── marker12_or_none/   # Binary: Touch (1 or 2) vs No-touch
│   │   ├── touch_detection_model.pkl
│   │   ├── model_performance.png
│   │   └── model_info.txt
│   │
│   ├── multiclass_marker/  # Multiclass: 0, 1, 2 classification
│   │   ├── multiclass_model.pkl
│   │   ├── multiclass_performance.png
│   │   └── model_info.txt
│   │
│   └── 3data_marker12_or_none/  # Fine-tuned binary model (NEW)
│       ├── 3data_marker12_or_none_model.pkl  # Best performing model
│       ├── model_comparison.png
│       └── model_info.txt
│
├── scripts/                # Training and inference scripts
│   ├── train_model.py              # Original training script
│   ├── train_model_marker1.py      # Marker 1 specific training
│   ├── train_model_multiclass.py   # Multiclass training
│   ├── finetune_marker12_or_none.py # Fine-tuning script
│   ├── inference.py                 # Basic inference
│   ├── inference_multiclass.py     # Multiclass inference
│   ├── inference_3data.py          # Inference for fine-tuned model
│   ├── test_detection.py           # Testing utilities
│   └── phantom_model.py            # Model integration
│
└── utils/                  # Utility modules
    ├── preprocess.py       # EEG preprocessing functions
    ├── features.py         # Feature extraction
    ├── utils.py           # General utilities
    └── check_markers.py    # Marker validation

```

## Model Performance Summary

### 3data_marker12_or_none (RECOMMENDED)
- **Type**: Binary classification (Touch vs No-touch)
- **Training Data**: 3 datasets combined
- **Accuracy**: 76.5%
- **ROC AUC**: 0.735
- **Touch Recall**: 85%
- **Use Case**: Best overall performance for touch detection

### marker12_or_none
- **Type**: Binary classification
- **Training Data**: Single dataset
- **Accuracy**: 69.6% (on original test set)
- **Use Case**: Baseline binary model

### marker1_or_not
- **Type**: Binary classification (Marker 1 only)
- **Accuracy**: ~60%
- **Use Case**: Specific marker 1 detection

### multiclass_marker
- **Type**: 3-class classification (0, 1, 2)
- **Accuracy**: Varies by class
- **Use Case**: When you need to distinguish between marker types

## Usage Examples

### Running Inference with Fine-tuned Model
```bash
cd scripts
python inference_3data.py --model 3data_marker12_or_none --file ../datasets/your_data.csv
```

### Fine-tuning on New Data
```bash
cd scripts
python finetune_marker12_or_none.py
```

### Training a New Model
```bash
cd scripts
python train_model.py
```

## Key Features
- **Sample Rate**: 125 Hz (configurable)
- **Window Size**: 125 samples (1 second)
- **Overlap**: 50%
- **Channel Pairs**: C3-C4, P3-P4, P7-P8, T7-T8
- **Frequency Bands**: Delta, Theta, Alpha, Beta, Gamma
- **Feature Count**: 160 per window

## Notes
- Models are saved as pickle files with embedded scalers
- All models use Random Forest classifiers
- SMOTE is used for class balancing during training
- Marker propagation: 24-49 samples before, 100-200 samples after