# Respiratory Health AI

Multimodal respiratory screening prototype built with Flask, PyTorch, and scikit-learn.

The system uses:

- a chest X-ray classifier (multi-disease deep learning model),
- an EHR/tabular lung cancer risk model,
- and score-level fusion to produce a single combined risk score.

---

## Table of Contents

1. [Project overview](#project-overview)
2. [Current capabilities](#current-capabilities)
3. [Architecture](#architecture)
4. [Tech stack](#tech-stack)
5. [Repository structure](#repository-structure)
6. [Datasets](#datasets)
7. [Models](#models)
8. [How inference works](#how-inference-works)
9. [Evaluation workflow](#evaluation-workflow)
10. [Installation and setup](#installation-and-setup)
11. [Running the app](#running-the-app)
12. [Training](#training)
13. [HTTP API](#http-api)
14. [Web pages](#web-pages)
15. [Testing](#testing)
16. [Important implementation notes](#important-implementation-notes)
17. [Limitations and disclaimer](#limitations-and-disclaimer)

---

## Project overview

This project is an end-to-end medical-AI prototype that combines two different modalities:

1. **Image modality**: chest X-ray image classification into five classes.
2. **Structured modality**: EHR-style tabular prediction of lung cancer risk.

Outputs from both are combined using a weighted average in `app.py` (`fuse_patient_risk`).

The app is built as a server-rendered Flask application with a lightweight static frontend (HTML/CSS/JS), no frontend build pipeline required.

---

## Current capabilities

- **X-ray prediction endpoint (`/predict`)**
  - Accepts JPG/PNG/JPEG chest X-ray
  - Returns predicted class, confidence, class probabilities
  - Returns `abnormal_probability` (sum of non-Normal class probabilities)
  - Returns Grad-CAM overlay as base64 image
  - Returns disease-specific recommendations

- **EHR risk endpoint (`/predict-lung-cancer`)**
  - Accepts JSON or form data
  - Validates fields against metadata saved in `models/lung_cancer_model_meta.json`
  - Returns probability, risk band, and recommendations

- **Combined endpoint (`/predict-combined`)**
  - Accepts image + EHR fields together
  - Runs both models
  - Produces fused risk score and combined summary

- **Model accuracy dashboard (`/model-accuracy`)**
  - Reads `evaluation_results/evaluation_report.json`
  - Visualizes confusion matrices, metric comparisons, and ROC curves
  - Can trigger re-evaluation using `/run-evaluation`

---

## Architecture

```text
Chest X-ray image ---> Multi-disease ResNet18 ------\
                                                     > weighted fusion ---> combined score + risk band
EHR features -------> sklearn pipeline (RF) --------/
```

Core runtime is in `app.py`:

- loads artifacts from `models/`
- exposes all Flask routes
- handles input validation
- performs inference and response formatting

---

## Tech stack

### Backend / serving

- Flask 3 (`Flask`, `Jinja2`, `Werkzeug`)

### Image model (deep learning)

- PyTorch (`torch`, `torchvision`)
- ResNet18-based model for 5-class chest X-ray classification
- Pillow + torchvision transforms for image preprocessing

### EHR model (tabular ML)

- scikit-learn pipeline
- `ColumnTransformer` + preprocessing + `RandomForestClassifier`
- `joblib`/pickle for serialization

### Data / utilities / plotting

- `numpy`, `pandas`, `scipy`
- `matplotlib`, `seaborn`

All dependency versions are pinned in `requirements.txt`.

---

## Repository structure

```text
pneumonia-detection/
├── app.py
├── README.md
├── requirements.txt
├── start.sh
├── .gitignore
├── datasets/
├── docs/
├── evaluation_results/
├── models/
├── src/
├── static/
├── templates/
└── tests/
```

### Key folders

- `src/`
  - `train_multi_disease.py`: trains 5-class X-ray model
  - `train_lung_cancer.py`: trains EHR lung-cancer model
  - `train.py`: legacy 2-class pneumonia model training
  - `comprehensive_evaluation.py`: unified evaluation + plotting
  - `evaluate.py`: legacy quick evaluator
  - `model.py`, `data_loader.py`: legacy helpers

- `models/`
  - `multi_disease_model.pth`
  - `multi_disease_meta.json`
  - `lung_cancer_model.pkl`
  - `lung_cancer_model_meta.json`
  - `model.pth` (legacy 2-class pneumonia fallback)

- `evaluation_results/`
  - `evaluation_report.json`
  - `results_table.csv`
  - `confusion_matrices.png`
  - `metrics_comparison.png`
  - `roc_curves.png`

- `templates/`
  - `landing.html`
  - `dashboard.html`
  - `how_it_works.html`
  - `accuracy.html`
  - `base.html`

- `static/`
  - `css/main.css`, `css/styles.css`, `css/accuracy.css`
  - `js/app.js`, `js/site.js`

- `tests/`
  - `test_dashboard.py`: smoke/integration checks for files, routes, fusion logic, and dataset class coverage

---

## Datasets

Expected under `datasets/`:

1. `datasets/chest_xray/` (Kaggle pneumonia dataset)
2. `datasets/COVID-19_Radiography_Dataset/`
3. `datasets/TB_Chest_Radiography_Database/`
4. `datasets/lung_cancer_dataset.csv`

### Multi-disease training classes

The 5-class X-ray model uses:

- `Normal`
- `Pneumonia`
- `COVID-19`
- `Tuberculosis`
- `Lung_Opacity`

`src/train_multi_disease.py` merges data from the above sources and applies a per-class cap (`--max-per-class`, default `1500`) before stratified split.

---

## Models

### 1) Multi-disease chest X-ray model

- Framework: PyTorch (`torchvision.models.resnet18`)
- Output classes: 5
- Artifact: `models/multi_disease_model.pth`
- Metadata: `models/multi_disease_meta.json`
- Inference function: `run_multi_disease_inference` in `app.py`
- Explainability: Grad-CAM from `layer4[-1]` via `generate_gradcam_overlay`

### 2) Legacy pneumonia model (fallback)

- 2-class (`Normal`, `Pneumonia`) ResNet18 head
- Artifact: `models/model.pth`
- Used when multi-disease artifacts are unavailable

### 3) EHR lung cancer risk model

- scikit-learn pipeline + RandomForest
- Artifact: `models/lung_cancer_model.pkl`
- Metadata/schema: `models/lung_cancer_model_meta.json`
- Inference function: `_run_lung_cancer_model` in `app.py`

---

## How inference works

### X-ray path (`POST /predict`)

1. validate image extension
2. load image and preprocess to 224x224
3. run model forward pass
4. compute top class + probabilities
5. compute `abnormal_probability`
6. generate Grad-CAM overlay
7. return JSON response

### EHR path (`POST /predict-lung-cancer`)

1. parse JSON/form payload
2. validate required features and allowed categorical values
3. run sklearn pipeline `predict_proba`
4. map probability to risk band
5. return risk output + recommendations

### Combined path (`POST /predict-combined`)

Uses both outputs and applies:

```python
FUSION_WEIGHTS = {
    "xray_abnormal": 0.5,
    "ehr_lung_cancer": 0.5,
}
```

Combined score is clipped and returned with a human-readable risk band.

---

## Evaluation workflow

Main script: `src/comprehensive_evaluation.py`

It evaluates:

1. `multi_disease` model
2. `pneumonia` legacy model
3. `lung_cancer` model

Then generates:

- metrics (accuracy/precision/recall/F1)
- confusion matrices
- ROC curves
- CSV summary table
- JSON report consumed by `accuracy.html`

Output location: `evaluation_results/`.

Detailed methodology and latest reported numbers are documented in
`docs/MODEL_ACCURACY_GUIDE.md`.

---

## Detailed ML project explanation

This section explains the complete machine-learning pipeline in one place: dataset sources, training flow, model metrics, and how score-level fusion changes the final decision output.

### End-to-end pipeline summary

1. collect and prepare image + tabular datasets under `datasets/`
2. train X-ray model(s) and EHR model using scripts in `src/`
3. save trained artifacts and metadata under `models/`
4. run inference through Flask (`app.py`) using API endpoints
5. evaluate performance with `src/comprehensive_evaluation.py`
6. optionally combine X-ray and EHR outputs using weighted fusion

### 1) Datasets used

#### Image datasets (for chest X-ray model)

- `datasets/chest_xray/` (Kaggle pneumonia-style Normal/Pneumonia data)
- `datasets/COVID-19_Radiography_Dataset/`
- `datasets/TB_Chest_Radiography_Database/`

The multi-disease image pipeline builds these classes:

- `Normal`
- `Pneumonia`
- `COVID-19`
- `Tuberculosis`
- `Lung_Opacity`

`src/train_multi_disease.py` merges samples across sources and applies a class cap (`--max-per-class`, default `1500`) to reduce imbalance before a stratified split.

#### Tabular dataset (for EHR lung cancer model)

- `datasets/lung_cancer_dataset.csv`

This file is used by `src/train_lung_cancer.py` to train a structured-risk model from patient-style variables (age, smoking exposure, clinical and environmental factors).

### 2) How model training works

#### A. Multi-disease chest X-ray training

- Script: `src/train_multi_disease.py`
- Backbone: ResNet18 (PyTorch / torchvision)
- Output: 5-class softmax
- Artifacts:
  - `models/multi_disease_model.pth`
  - `models/multi_disease_meta.json`

Training includes image transforms, class balancing logic, stratified train/validation split, and metric tracking per epoch.

#### B. Legacy 2-class pneumonia model (fallback)

- Script: `src/train.py`
- Classes: `Normal`, `Pneumonia`
- Artifact: `models/model.pth`

This model is loaded only when multi-disease artifacts are unavailable.

#### C. EHR lung cancer risk model

- Script: `src/train_lung_cancer.py`
- Pipeline: preprocessing (`ColumnTransformer`) + `RandomForestClassifier`
- Artifacts:
  - `models/lung_cancer_model.pkl`
  - `models/lung_cancer_model_meta.json`

The metadata file stores expected schema and allowed categorical values for runtime validation.

### 3) Inference outputs and meanings

#### X-ray model output (`/predict`)

- top predicted class + confidence
- per-class probabilities
- `abnormal_probability`: sum of all non-`Normal` class probabilities
- Grad-CAM overlay for interpretability

#### EHR model output (`/predict-lung-cancer`)

- lung-cancer probability (`predict_proba`)
- risk band mapping
- recommendation text

### 4) Model accuracy and performance (latest recorded values)

Based on generated evaluation artifacts (`evaluation_results/results_table.csv` and `evaluation_results/evaluation_report.json`):

- **Multi-Disease X-ray model**
  - Accuracy: **92.39%**
  - Precision: **93.38%**
  - Recall: **92.97%**
  - F1: **93.08%**
  - Samples: **1340**

- **Legacy Pneumonia model**
  - Accuracy: **92.72%**
  - Precision: **93.90%**
  - Recall: **96.54%**
  - F1: **95.20%**
  - Samples: **1044**

- **Lung Cancer EHR model**
  - Accuracy: **69.82%**
  - Precision: **77.93%**
  - Recall: **78.25%**
  - F1: **78.09%**
  - Samples: **10000**

### 5) Fusion logic: without fusion vs with fusion

#### Without fusion (single-modality decisions)

- X-ray endpoint uses only image evidence.
- EHR endpoint uses only tabular patient data.
- Each model produces an independent risk estimate for its own task context.

This mode is useful when only one modality is available or when you want to inspect each model separately.

#### With fusion (combined endpoint)

`/predict-combined` runs both models and fuses their scalar outputs:

```python
FUSION_WEIGHTS = {
    "xray_abnormal": 0.5,
    "ehr_lung_cancer": 0.5,
}
```

Combined score is a weighted average of:

- X-ray `abnormal_probability`
- EHR lung-cancer probability

Then mapped to a combined risk band.

#### Practical interpretation

- **Without fusion**: you get modality-specific predictions (more separable and easier to debug per model).
- **With fusion**: you get a single integrated score that considers both image abnormalities and EHR risk factors together.

Important: current fusion is a **heuristic score-level blend**, not a separately trained multimodal classifier with paired ground-truth labels.

### 5.1) Class-wise X-ray performance (including Tuberculosis)

From the latest `evaluation_results/evaluation_report.json` confusion matrix (multi-disease model):

| Class | Support | Precision | Recall | F1 |
| ----- | ------- | --------- | ------ | --- |
| Normal | 300 | 82.18% | 90.67% | 86.21% |
| Pneumonia | 300 | 95.82% | 99.33% | 97.55% |
| COVID-19 | 300 | 95.73% | 89.67% | 92.60% |
| Tuberculosis | 140 | 99.28% | 97.86% | 98.56% |
| Lung_Opacity | 300 | 93.91% | 87.33% | 90.50% |

**Tuberculosis performance is strong** in this evaluation run:

- Precision: **99.28%** (very few false TB alarms)
- Recall: **97.86%** (most true TB samples were detected)
- F1-score: **98.56%**

### 6) How to evaluate and compare performance

Run:

```bash
python src/comprehensive_evaluation.py
```

This regenerates:

- confusion matrices
- ROC curves
- metrics comparison chart
- CSV table + JSON report

Dashboard route: `/model-accuracy`

Use this to compare individual model behavior and monitor system-level summary trends over time.

---

## Installation and setup

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the app

```bash
python app.py
```

or:

```bash
chmod +x start.sh
./start.sh
```

Open:

- `http://127.0.0.1:5000/`
- `http://127.0.0.1:5000/dashboard`
- `http://127.0.0.1:5000/model-accuracy`

---

## Training

### Train multi-disease X-ray model

```bash
python src/train_multi_disease.py --epochs 3 --max-per-class 1500
```

### Train legacy pneumonia model

```bash
python src/train.py
```

### Train lung cancer EHR model

```bash
python src/train_lung_cancer.py
```

---

## HTTP API

### GET routes

- `/`
- `/dashboard`
- `/how-it-works`
- `/model-accuracy`
- `/evaluation-image/<path:filename>`

### POST routes

- `/predict`
  - body: `multipart/form-data` with `image`
- `/predict-lung-cancer`
  - body: JSON or form fields
- `/predict-combined`
  - body: `multipart/form-data` with `image` + EHR fields
- `/run-evaluation`
  - triggers `src/comprehensive_evaluation.py`

### Example: X-ray prediction

```bash
curl -F "image=@datasets/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg" \
  http://127.0.0.1:5000/predict
```

### Example: EHR prediction

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"age":62,"gender":"Male","pack_years":35,"radon_exposure":"Low","asbestos_exposure":"No","secondhand_smoke_exposure":"No","copd_diagnosis":"No","alcohol_consumption":"Moderate","family_history":"No"}' \
  http://127.0.0.1:5000/predict-lung-cancer
```

---

## Web pages

- `/` -> landing page (`templates/landing.html`)
- `/dashboard` -> tabbed workflow: chest X-ray, optional EHR risk form, combined fusion (`templates/dashboard.html`)
- `/how-it-works` -> architecture + explanation (`templates/how_it_works.html`)
- `/model-accuracy` -> evaluation dashboard (`templates/accuracy.html`)

---

## Testing

Run:

```bash
python tests/test_dashboard.py
```

Checks include:

- required file availability
- Flask route registration
- fusion logic correctness and clamping
- multi-disease dataset class coverage validation

---

## Important implementation notes

- The frontend script `static/js/app.js` contains logic for X-ray, EHR, and combined tabs.
- `templates/dashboard.html` includes **Chest X-ray**, **EHR risk**, and **Combined** tabs (see `templates/partials/ehr_fields.html`).
- EHR and combined model functionality are fully available through API endpoints and backend routes.
- `run-evaluation` executes a subprocess with timeout to regenerate `evaluation_results/`.

---

## Limitations and disclaimer

- Educational/research prototype only; not a medical device.
- Not intended for real-world diagnosis, triage, or treatment decisions.
- Fusion score is a heuristic weighted blend, not a clinically validated endpoint.
- No paired patient-level multimodal ground-truth dataset is used for end-to-end fusion training.
- Dataset distribution shift can significantly change real-world performance.
- Grad-CAM is an interpretability aid only.

---

For deeper implementation notes, see `docs/`.
