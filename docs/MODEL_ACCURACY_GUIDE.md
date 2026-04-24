# Model Accuracy Dashboard - User Guide

## Overview
The Model Accuracy Dashboard provides comprehensive performance metrics and visualizations for both AI models in the Respiratory Health AI system.

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

## Features

### 1. **Quick Metrics Overview**
View at-a-glance accuracy metrics:
- Pneumonia Detection Model Accuracy
- Lung Cancer Risk Model Accuracy
- Combined Average Accuracy
- Total Samples Evaluated

### 2. **Detailed Performance Table**
Comprehensive breakdown including:
- Accuracy
- Precision
- Recall
- F1-Score
- Sample sizes

### 3. **Visual Analytics**
Interactive visualizations:
- **Confusion Matrices**: Shows prediction accuracy breakdown (True Positives, False Positives, etc.)
- **Metrics Comparison**: Side-by-side bar chart comparing all performance metrics
- **ROC Curves**: Receiver Operating Characteristic curves showing model discrimination ability

### 4. **Confusion Matrix Breakdown**
Detailed view of:
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)
- True Positives (TP)

### 5. **Run Evaluation**
Click the **"Run Model Evaluation"** button to:
- Re-evaluate both models on fresh data
- Generate new metrics and graphs
- Update all visualizations

**Note:** Evaluation takes approximately 1-2 minutes to complete.

## What Gets Evaluated

### Pneumonia Detection Model
- **Dataset**: Chest X-ray images (validation set)
- **Classes**: Normal vs Pneumonia
- **Method**: ResNet18 deep learning model

### Lung Cancer Risk Model
- **Dataset**: Electronic Health Records (EHR)
- **Classes**: No Cancer vs Lung Cancer
- **Method**: RandomForest classifier on patient data

## Understanding the Metrics

### Accuracy
Percentage of correct predictions out of all predictions.

### Precision
Of all positive predictions, how many were actually positive?
- Higher precision = fewer false alarms

### Recall (Sensitivity)
Of all actual positive cases, how many did we catch?
- Higher recall = fewer missed cases

### F1-Score
Harmonic mean of precision and recall.
- Balanced measure when both are important

### Confusion Matrix
- **True Positive (TP)**: Correctly predicted positive
- **True Negative (TN)**: Correctly predicted negative
- **False Positive (FP)**: Incorrectly predicted positive
- **False Negative (FN)**: Incorrectly predicted negative

## Technical Details

### Files Generated
When you run evaluation, the following files are created in `evaluation_results/`:
- `confusion_matrices.png` - Heatmap visualizations
- `metrics_comparison.png` - Bar chart comparison
- `roc_curves.png` - ROC curve visualizations
- `results_table.csv` - Metrics in CSV format
- `evaluation_report.json` - Complete results in JSON

### Command Line Alternative
You can also run evaluation from the command line:
```bash
python comprehensive_evaluation.py
```

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

## Current Performance (Last Evaluation)

Based on the most recent evaluation:

### Pneumonia Detection Model
- **Accuracy**: 93.58%
- **Precision**: 93.97%
- **Recall**: 97.52%
- **F1-Score**: 95.71%
- **Samples**: 1,044 chest X-rays

### Lung Cancer Risk Model
- **Accuracy**: 69.82%
- **Precision**: 77.93%
- **Recall**: 78.25%
- **F1-Score**: 78.09%
- **Samples**: 10,000 patient records

### Combined System
- **Average Accuracy**: 81.70%
- **Weighted Accuracy**: 72.07%

## Next Steps

1. **Regular Monitoring**: Re-run evaluation periodically to track model performance
2. **Data Updates**: When new data is added, re-evaluate to see impact
3. **Model Improvements**: Use metrics to identify areas for enhancement
4. **Documentation**: Share results with stakeholders using generated CSV/JSON files

## Support

For issues or questions:
1. Check the evaluation logs in terminal
2. Verify both model files exist (`model.pth`, `lung_cancer_model.pkl`)
3. Ensure all dependencies are installed (`pip install -r requirements.txt`)
