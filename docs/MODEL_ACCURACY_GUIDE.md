# Model Accuracy Dashboard - Evaluation Method and Results

## Overview

This document explains how Respiratory Health AI is evaluated, what metrics are used, and what the latest measured results are.

## How to Access

### Method 1: Web Interface (Recommended)
1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to: `http://127.0.0.1:5000`

3. Click the **"Model Accuracy"** button in the top right corner of the page

### Method 2: Direct URL
Navigate directly to: `http://127.0.0.1:5000/model-accuracy`

## What is evaluated

### 1) Multi-disease chest X-ray model

- Model: ResNet18 (5-class)
- Classes: `Normal`, `Pneumonia`, `COVID-19`, `Tuberculosis`, `Lung_Opacity`
- Validation split: stratified holdout (`test_size=0.2`)
- Key outputs: class predictions, confusion matrix, ROC (Normal vs Abnormal), AUC

### 2) Legacy pneumonia model (fallback)

- Model: ResNet18 (2-class)
- Classes: `Normal`, `Pneumonia`
- Validation split: loader-based validation set used by legacy pipeline
- Key outputs: binary confusion matrix, ROC, AUC

### 3) EHR lung cancer risk model

- Model: scikit-learn tabular pipeline (`RandomForestClassifier` + preprocessing)
- Labels: `No Cancer`, `Lung Cancer`
- Validation split: stratified holdout (`test_size=0.2`)
- Key outputs: binary confusion matrix, ROC, AUC

### 4) Combined/fusion proxy metric

- Current fusion weights:
  - `xray_abnormal = 0.5`
  - `ehr_lung_cancer = 0.5`
- If both modalities are present:
  - fusion proxy = weighted average of X-ray accuracy and EHR accuracy
- If one modality is unavailable:
  - fallback proxy uses available modality (`xray_only` or `ehr_only`)

Important: this is a score-level proxy for system summary, not a separately trained paired multimodal classifier.

## Evaluation metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC curve and AUC

These are computed in `src/comprehensive_evaluation.py` for each model.

## Technical Details

### Files generated

When evaluation runs, the following files are written in `evaluation_results/`:
- `confusion_matrices.png` - Heatmap visualizations
- `metrics_comparison.png` - Bar chart comparison
- `roc_curves.png` - ROC curve visualizations
- `results_table.csv` - Metrics in CSV format
- `evaluation_report.json` - canonical metrics source for the dashboard

### Running evaluation

From CLI:
```bash
python src/comprehensive_evaluation.py
```

From web UI:

1. open `/model-accuracy`
2. click **Run Model Evaluation**

## Troubleshooting

### "No evaluation results found"
- Click "Run Model Evaluation" button
- Or run: `python comprehensive_evaluation.py`

### Graphs not displaying
- Ensure evaluation has been run at least once
- Check that `evaluation_results/` folder contains PNG files
- Try refreshing the page

### Evaluation takes too long
- Normal: 1-2 minutes for full evaluation
- Check terminal for progress messages
- Ensure both models are properly loaded

## Latest results (from `evaluation_results/evaluation_report.json`)

### Multi-disease chest X-ray model

- Accuracy: **92.39%**
- Precision: **93.38%**
- Recall: **92.97%**
- F1-score: **93.08%**
- Samples: **1,340**

### Legacy pneumonia model

- Accuracy: **94.06%**
- Precision: **95.73%**
- Recall: **96.46%**
- F1-score: **96.10%**
- Samples: **1,044**

### EHR lung cancer risk model

- Accuracy: **69.82%**
- Precision: **77.93%**
- Recall: **78.25%**
- F1-score: **78.09%**
- Samples: **10,000**

### Combined summary

- Average accuracy: **85.42%**
- Weighted accuracy: **74.31%**
- Fusion proxy (multimodal): **81.10%**

## Notes and limitations

- Evaluation is run at model level; fusion is currently a score-level proxy.
- Reported numbers are sensitive to dataset composition and split seed.
- This is an educational/research prototype and not a clinical diagnostic device.
