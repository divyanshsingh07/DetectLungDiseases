# Respiratory Health AI

> **MULTIMODAL PATIENT-LEVEL LUNG DISEASE STRATIFICATION USING EHR AND CHEST X-RAYS**
>
> An end-to-end Flask web application that combines a deep-learning chest X-ray
> classifier with an EHR-based lung cancer risk model. Two specialised models
> are run on two different patient inputs (an X-ray image and a structured
> patient record), and their outputs are fused at score-level into a single,
> interpretable respiratory health rating.

---

## Table of Contents

1. [Quick start](#quick-start)
2. [What this project is](#what-this-project-is)
3. [Key capabilities](#key-capabilities)
4. [System architecture](#system-architecture)
5. [Datasets](#datasets)
6. [Models](#models)
7. [Score-level fusion](#score-level-fusion)
8. [Evaluation results](#evaluation-results)
9. [Project structure](#project-structure)
10. [Installation](#installation)
11. [Training the models](#training-the-models)
12. [Running the app](#running-the-app)
13. [HTTP API](#http-api)
14. [Web UI pages](#web-ui-pages)
15. [Limitations and disclaimer](#limitations-and-disclaimer)

---

## Quick start

The fastest way to get the app running on `http://127.0.0.1:5000`:

### macOS / Linux

```bash
# 1. Clone & enter
git clone <repo-url> pneumonia-detection
cd pneumonia-detection

# 2. Create + activate a virtualenv
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Start the app
python app.py
#    or, with the helper script:
chmod +x start.sh && ./start.sh
```

### Windows (PowerShell)

```powershell
git clone <repo-url> pneumonia-detection
cd pneumonia-detection

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt

python app.py
```

Then open <http://127.0.0.1:5000>.

> Trained model artefacts (`models/*.pth`, `models/*.pkl`) are checked in,
> so the app works **without training or downloading any datasets**.

---

## What this project is

Respiratory Health AI is a **multimodal lung-disease screening prototype**.
A clinician (or researcher) can:

- Upload a chest X-ray image and get a 5-class disease prediction
  (Normal / Pneumonia / COVID-19 / Tuberculosis / Lung Opacity), plus a
  Grad-CAM visualisation of where the model "looked".
- Submit a small EHR record (age, smoking pack-years, radon exposure, COPD
  status, etc.) and get a lung cancer risk probability with a tier label and
  recommendations.
- Submit **both at the same time** to receive a *fused* respiratory health
  score that blends the imaging and the structured clinical signals.

This implements a **multi-model, late-fusion architecture** rather than a
single end-to-end multimodal network — the two modalities are handled by two
specialised models, and only their final probabilities are combined.

---

## Key capabilities

| Capability                       | Backed by                                                          |
| -------------------------------- | ------------------------------------------------------------------ |
| Multi-disease X-ray screening    | ResNet18 fine-tuned on 6,700 chest X-rays from 3 public datasets   |
| Lung cancer risk from EHR        | scikit-learn `RandomForestClassifier` pipeline trained on 50K rows |
| Patient-level fusion             | Weighted-average score blending both model outputs                 |
| Explainability                   | Grad-CAM heatmap overlay for X-ray predictions                     |
| Reproducible evaluation          | One-button run that regenerates confusion matrices, ROC curves,    |
|                                  | metric bars, CSV table, and JSON report                            |
| Minimal product UI               | Dark-themed, focused upload-first interface (Plus Jakarta Sans /   |
|                                  | Playfair Display typography)                                       |

---

## System architecture

```text
                ┌──────────────────────┐        ┌──────────────────────┐
   Chest X-ray  │  Multi-disease       │        │  Lung cancer risk    │  EHR record
  ────────────▶ │  ResNet18 (5-class)  │        │  RandomForest (sklearn)│ ◀────────────
   PNG/JPG      │  + Grad-CAM          │        │  + StandardScaler +  │  age, pack-years,
                │                      │        │  OneHotEncoder       │  COPD, radon, ...
                └─────────┬────────────┘        └──────────┬───────────┘
                          │ P(abnormal) ∈ [0,1]            │ P(cancer) ∈ [0,1]
                          └────────────┬───────────────────┘
                                       ▼
                          ┌──────────────────────────┐
                          │  Score-level fusion      │
                          │  0.5·P(abn) + 0.5·P(can) │
                          └─────────────┬────────────┘
                                        ▼
                       Combined respiratory health score (%)
                       + risk band (Low / Moderate / High / Very High)
                       + per-model recommendations
```

All inference happens server-side inside `app.py`. The Flask layer is the
only piece talking to the browser; the front-end is plain HTML/CSS/JS with
no build step.

---

## Datasets

Three public chest-X-ray datasets and one synthetic-style EHR CSV are used.
The image datasets are large (multiple GB) and are **not committed to the
repository** — they live under `datasets/` on the local filesystem.

### 1. Kaggle Chest X-Ray Pneumonia (Mooney 2018)

- Path: `datasets/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/`
- Used as one of three sources for `Normal` and `Pneumonia`
- Also serves the validation loader for the legacy 2-class model

### 2. COVID-19 Radiography Database (Rahman et al., Kaggle)

- Path: `datasets/COVID-19_Radiography_Dataset/`
- Sub-folders consumed:
  - `Normal/images` → contributes to `Normal`
  - `Viral Pneumonia/images` → contributes to `Pneumonia`
  - `COVID/images` → maps to `COVID-19`
  - `Lung_Opacity/images` → maps to `Lung_Opacity`
  - `*/masks/` directories are explicitly skipped

### 3. TB Chest Radiography Database (Kaggle)

- Path: `datasets/TB_Chest_Radiography_Database/`
- `Normal/` → contributes to `Normal`
- `Tuberculosis/` → maps to `Tuberculosis`

### 4. Lung Cancer EHR dataset

- Path: `datasets/lung_cancer_dataset.csv`
- 50,000 rows, 9 features + target
- Numeric: `age`, `pack_years`
- Categorical: `gender`, `radon_exposure`, `asbestos_exposure`,
  `secondhand_smoke_exposure`, `copd_diagnosis`, `alcohol_consumption`,
  `family_history`
- Target: `lung_cancer` ∈ {`Yes`, `No`}

### Class composition for the multi-disease classifier

The fusion step caps each class at `--max-per-class` (default **1,500**) so
the much larger `Normal` pool doesn't dominate the gradient. The final
training set used:

| Class        | Sources merged                                                     | Cap (used) |
| ------------ | ------------------------------------------------------------------ | ---------- |
| Normal       | `chest_xray/train/NORMAL` + COVID-19 `Normal/images` + TB `Normal` | 1,500      |
| Pneumonia    | `chest_xray/train/PNEUMONIA` + COVID-19 `Viral Pneumonia/images`   | 1,500      |
| COVID-19     | COVID-19 `COVID/images`                                            | 1,500      |
| Tuberculosis | TB `Tuberculosis/`                                                 | 700 (max available after dedup) |
| Lung_Opacity | COVID-19 `Lung_Opacity/images`                                     | 1,500      |
| **Total**    |                                                                    | **6,700** |

Stratified 80 / 20 split → **5,360** train / **1,340** validation samples.

---

## Models

### Model 1 — Multi-disease chest X-ray classifier

| Item              | Value                                                          |
| ----------------- | -------------------------------------------------------------- |
| Architecture      | `torchvision.models.resnet18` with ImageNet pretrained weights |
| Trainable layers  | `layer4` and `fc` (rest frozen for fast transfer learning)     |
| Output head       | `nn.Linear(512, 5)`                                            |
| Input             | 224×224 RGB, ImageNet mean/std normalised                      |
| Train augment     | RandomHorizontalFlip, RandomRotation(10°), ColorJitter         |
| Loss              | `CrossEntropyLoss(weight=inverse-frequency)` — handles class imbalance |
| Optimiser         | Adam (lr = 1e-3)                                               |
| Batch size        | 32                                                             |
| Epochs            | 3                                                              |
| Best val accuracy | **92.39 %**                                                    |
| Saved as          | `models/multi_disease_model.pth` + `models/multi_disease_meta.json` |
| Visualisation     | Grad-CAM hook on `layer4[-1]` overlaid on the original image   |

Trainer: `src/train_multi_disease.py`. The script also `validate_class_counts(...)`
to fail early if any disease class doesn't have enough images for a stratified
split.

### Model 2 — Legacy 2-class pneumonia model

| Item             | Value                                                             |
| ---------------- | ----------------------------------------------------------------- |
| Architecture     | ResNet18 (frozen body, only `fc` retrained)                       |
| Output head      | `nn.Linear(512, 2)` — Normal vs Pneumonia                         |
| Saved as         | `models/model.pth`                                                |
| Used for         | Fallback if multi-disease model is missing; legacy benchmark      |
| Val accuracy     | **92.72 %**, recall **96.54 %**                                   |

Trainer: `src/train.py` (with `src/data_loader.py` for the Kaggle pneumonia
dataset).

### Model 3 — Lung cancer EHR classifier

| Item              | Value                                                                   |
| ----------------- | ----------------------------------------------------------------------- |
| Pipeline          | `ColumnTransformer(StandardScaler + OneHotEncoder)` → `RandomForestClassifier` |
| `RandomForest`    | 300 trees, `min_samples_leaf=2`, `class_weight="balanced"`              |
| Train / test      | 40,000 / 10,000 stratified split (`random_state=42`)                    |
| Test accuracy     | **69.82 %** · Test ROC-AUC **0.7288**                                   |
| Saved as          | `models/lung_cancer_model.pkl` + `models/lung_cancer_model_meta.json`   |
| Top features      | `pack_years` (45.1 %) · `age` (32.1 %) · `radon_exposure_High` (2.5 %)  |
| Risk bands        | <25 % Low · 25–55 % Moderate · 55–80 % High · ≥80 % Very High           |

Trainer: `src/train_lung_cancer.py`. The metadata JSON also stores the allowed
values for each categorical feature, which the Flask layer uses to build the
form schema and validate every incoming request.

---

## Score-level fusion

Both the X-ray model and the EHR model produce a probability of "something
abnormal". Those two probabilities are blended into a single patient-level
score using a weighted average:

```python
# app.py
FUSION_WEIGHTS = {
    "xray_abnormal":   0.5,
    "ehr_lung_cancer": 0.5,
}

def fuse_patient_risk(xray_result, lung_result):
    p_xray = clip(xray_result["abnormal_probability"] / 100, 0, 1)
    p_ehr  = clip(lung_result["probability"] / 100, 0, 1)
    score  = 0.5 * p_xray + 0.5 * p_ehr           # weighted average
    return {
        "score":     round(score * 100, 2),
        "risk_band": risk_band(score),             # Low / Moderate / High / Very High
        "method":    "weighted_average",
        "weights":   {...},
        "components": {"xray_abnormal": ..., "ehr_lung_cancer": ...},
    }
```

`abnormal_probability` for the X-ray side is the sum of the model's
probabilities for every non-Normal class (Pneumonia + COVID-19 + Tuberculosis +
Lung_Opacity). The result is clipped to `[0, 100] %` for safety. The fusion
weights (`0.5 / 0.5`) are configurable in one place in `app.py`.

> **Why not a single end-to-end multimodal model?**
> The project does not include a paired patient-level dataset that links each
> X-ray to one specific EHR record with a fused ground-truth label. Without
> such a dataset, training a joint network is not meaningful. Score-level
> ("late") fusion is the appropriate engineering choice in this scenario and
> is exactly what the Metrics page makes explicit.

---

## Evaluation results

Generated by `src/comprehensive_evaluation.py` and stored in
`evaluation_results/`. Click **Re-run evaluation** on `/model-accuracy` to
regenerate everything in one go.

| Model                        | Accuracy | Precision | Recall | F1     | Samples |
| ---------------------------- | -------: | --------: | -----: | -----: | ------: |
| Multi-Disease X-ray (5-class)| 0.9239   | 0.9338    | 0.9297 | 0.9308 | 1,340   |
| Pneumonia (legacy 2-class)   | 0.9272   | 0.9390    | 0.9654 | 0.9520 | 1,044   |
| Lung Cancer Risk (EHR)       | 0.6982   | 0.7793    | 0.7825 | 0.7809 | 10,000  |
| Combined (macro average)     | 0.8498   | 0.8840    | 0.8925 | 0.8879 | 12,384  |
| Combined (weighted by samples)| 0.7419  | —         | —      | —      | 12,384  |

### Multi-disease confusion matrix (validation set, 1,340 X-rays)

|              | Normal | Pneumonia | COVID-19 | Tuberculosis | Lung_Opacity |
| ------------ | -----: | --------: | -------: | -----------: | -----------: |
| Normal       |    272 |        12 |        5 |            0 |           11 |
| Pneumonia    |      2 |       298 |        0 |            0 |            0 |
| COVID-19     |     24 |         1 |      269 |            1 |            5 |
| Tuberculosis |      1 |         0 |        1 |          137 |            1 |
| Lung_Opacity |     32 |         0 |        6 |            0 |          262 |

Charts produced by the same run (served at `/evaluation-image/<file>`):

- `confusion_matrices.png` — heatmap per model
- `metrics_comparison.png` — accuracy/precision/recall/F1 bar chart
- `roc_curves.png` — Normal-vs-Abnormal ROC for each model
- `results_table.csv` — flat table for spreadsheets
- `evaluation_report.json` — machine-readable, consumed by the Metrics page

---

## Project structure

```text
pneumonia-detection/
├── app.py                          # Flask entry point — all inference routes
├── start.sh                        # `python app.py` wrapper with summary banner
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/                            # All Python source code
│   ├── data_loader.py              # Kaggle pneumonia ImageFolder loader
│   ├── model.py                    # Tiny ResNet18 factory (legacy 2-class)
│   ├── train.py                    # Trains the legacy 2-class model
│   ├── train_lung_cancer.py        # Trains the EHR RandomForest pipeline
│   ├── train_multi_disease.py      # Trains the 5-class multi-disease model
│   ├── evaluate.py                 # Quick CLI eval for the legacy model
│   └── comprehensive_evaluation.py # Full reporting (used by the Metrics page)
│
├── models/                         # All trained artefacts
│   ├── model.pth                   # Legacy 2-class pneumonia ResNet18
│   ├── multi_disease_model.pth     # 5-class multi-disease ResNet18
│   ├── multi_disease_meta.json     # Class names, training history, metrics
│   ├── lung_cancer_model.pkl       # sklearn pipeline (preprocess + RF)
│   └── lung_cancer_model_meta.json # Feature schema + allowed values + metrics
│
├── datasets/                       # Raw data (not committed)
│   ├── chest_xray/                 # Kaggle pneumonia (NORMAL / PNEUMONIA)
│   ├── COVID-19_Radiography_Dataset/
│   ├── TB_Chest_Radiography_Database/
│   └── lung_cancer_dataset.csv     # 50k-row EHR table
│
├── tests/
│   └── test_dashboard.py           # Verifies routes, files, fusion logic
│
├── docs/                           # Long-form notes on the implementation
│   ├── COMBINED_VISUALIZATIONS_GUIDE.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── MODEL_ACCURACY_GUIDE.md
│   └── IMAGE_FIX_SUMMARY.md
│
├── evaluation_results/             # Output of comprehensive_evaluation.py
│   ├── evaluation_report.json
│   ├── results_table.csv
│   ├── confusion_matrices.png
│   ├── metrics_comparison.png
│   └── roc_curves.png
│
├── templates/                      # Jinja2 templates
│   ├── base.html                   # Site shell, nav, fonts, footer
│   ├── landing.html                # `/` — Playfair Display hero
│   ├── dashboard.html              # `/dashboard` — single upload-first card
│   ├── how_it_works.html           # `/how-it-works` — 3-card explainer
│   └── accuracy.html               # `/model-accuracy` — metrics + charts
│
└── static/
    ├── css/
    │   ├── main.css                # Site shell, navbar, landing, How-it-works
    │   ├── styles.css              # Dashboard / upload card / result panel
    │   └── accuracy.css            # Metrics page (cards, charts, banners)
    └── js/
        ├── site.js                 # Mobile drawer toggle
        └── app.js                  # Upload, prediction, render helpers
```

---

## Installation

```bash
# 1. Clone the repo and cd into it
cd pneumonia-detection

# 2. Create a virtualenv (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt
```

`requirements.txt` pins:

- `torch==2.4.1`, `torchvision==0.19.1` — image model
- `scikit-learn==1.3.2`, `pandas==2.0.3`, `scipy==1.10.1`, `joblib==1.3.2` — EHR pipeline
- `Flask==3.0.3`, `Jinja2>=3.1`, `Werkzeug>=3.0` — web layer
- `matplotlib==3.7.5`, `seaborn==0.13.2` — evaluation charts
- `pillow==10.4.0`, `numpy==1.24.4`, `requests>=2.31.0` — utilities

### Datasets

Trained model files (`models/*.pth`, `*.pkl`) ship with the repo, so you can
run the app immediately. To **re-train** or **re-evaluate**, place the raw
datasets under `datasets/` using the layout listed in
[Datasets](#datasets) above.

---

## Training the models

All scripts use absolute paths derived from the project root, so they can be
run from anywhere:

```bash
# Multi-disease 5-class X-ray classifier
python3 src/train_multi_disease.py --epochs 3 --max-per-class 1500
#   → models/multi_disease_model.pth + models/multi_disease_meta.json

# Legacy 2-class pneumonia model
python3 src/train.py
#   → models/model.pth

# Lung cancer EHR classifier
python3 src/train_lung_cancer.py
#   → models/lung_cancer_model.pkl + models/lung_cancer_model_meta.json
```

Useful flags for `train_multi_disease.py`:

| Flag                | Default | Purpose                                            |
| ------------------- | ------: | -------------------------------------------------- |
| `--epochs`          |       3 | Training epochs                                    |
| `--batch-size`      |      32 | Mini-batch size                                    |
| `--max-per-class`   |    1500 | Cap per disease class (keeps gradient balanced)    |
| `--lr`              |    1e-3 | Adam learning rate                                 |
| `--workers`         |       2 | DataLoader workers                                 |
| `--seed`            |      42 | Reproducibility                                    |

Device selection is automatic in this priority order: **CUDA → Apple MPS → CPU**.

---

## Running the app

### 1. Start the Flask server

```bash
# from the project root, with the venv activated
python app.py
```

Equivalent helpers:

```bash
./start.sh                      # macOS / Linux convenience wrapper
python3 -m flask --app app run  # plain Flask CLI (no banner)
```

Default URL: **<http://127.0.0.1:5000>** — opens the landing page.

### 2. Open the pages

| URL                                        | What it is                                                  |
| ------------------------------------------ | ----------------------------------------------------------- |
| <http://127.0.0.1:5000>                    | Landing — Playfair Display hero, two CTAs                   |
| <http://127.0.0.1:5000/dashboard>          | Upload an X-ray, get a prediction + confidence + Grad-CAM   |
| <http://127.0.0.1:5000/how-it-works>       | 3-card explainer (X-ray, EHR, fusion)                       |
| <http://127.0.0.1:5000/model-accuracy>     | Metrics + charts + **Re-run evaluation** button             |

### 3. Run on a different host or port

```bash
# Bind to all interfaces on port 8080
python -m flask --app app run --host 0.0.0.0 --port 8080

# Or via env vars (works with `python app.py` too)
FLASK_RUN_PORT=8080 python app.py
```

### 4. Run in background / stop

```bash
# Run detached, send logs to a file
nohup python app.py > flask.log 2>&1 &
echo $! > flask.pid

# Tail the logs
tail -f flask.log

# Stop it again
kill "$(cat flask.pid)" && rm flask.pid

# Force-kill anything still on port 5000
lsof -ti :5000 | xargs kill -9 2>/dev/null
```

### 5. Re-train any model (optional)

```bash
python src/train_multi_disease.py --epochs 3 --max-per-class 1500
python src/train.py
python src/train_lung_cancer.py
```

See [Training the models](#training-the-models) for all flags. Datasets must
already be present under `datasets/` (see [Datasets](#datasets)).

### 6. Re-run the full evaluation

```bash
# CLI — regenerates evaluation_results/*.png and evaluation_report.json
python src/comprehensive_evaluation.py

# UI — same script, triggered from the Metrics page
# 1. open http://127.0.0.1:5000/model-accuracy
# 2. click "Re-run evaluation"

# API — same script, called over HTTP
curl -X POST http://127.0.0.1:5000/run-evaluation
```

### 7. Verification script

```bash
python tests/test_dashboard.py
```

Checks that every critical file exists, the Flask app imports cleanly,
every route is registered, the fusion math is correct (including clamping),
and every multi-disease class has enough images for a stratified split.

### 8. Sanity-check from the shell

```bash
# Is the server up?
curl -I http://127.0.0.1:5000

# Quick X-ray inference smoke test
curl -F "image=@datasets/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg" \
     http://127.0.0.1:5000/predict | python -m json.tool
```

---

## HTTP API

| Method | Path                                | Body / params                                                        | Returns                                                                          |
| ------ | ----------------------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| GET    | `/`                                 | —                                                                    | Landing page                                                                     |
| GET    | `/dashboard`                        | —                                                                    | Upload-first product UI                                                          |
| GET    | `/how-it-works`                     | —                                                                    | 3-card explainer                                                                 |
| GET    | `/model-accuracy`                   | —                                                                    | Metrics dashboard (renders `evaluation_report.json`)                             |
| GET    | `/evaluation-image/<filename>`      | —                                                                    | Serves PNGs from `evaluation_results/`                                           |
| POST   | `/predict`                          | `multipart/form-data`: `image` (PNG/JPG/JPEG)                        | Multi-disease prediction + class probabilities + Grad-CAM (base64)               |
| POST   | `/predict-lung-cancer`              | JSON or form: 9 EHR fields (see meta)                                | Probability + risk band + recommendations                                        |
| POST   | `/predict-combined`                 | `multipart/form-data`: `image` + 9 EHR fields                        | X-ray prediction + EHR risk + fused score with weights and components            |
| POST   | `/run-evaluation`                   | —                                                                    | Runs `src/comprehensive_evaluation.py` and returns the fresh JSON report         |

### Example requests

```bash
# X-ray only
curl -F "image=@chest.jpg" http://127.0.0.1:5000/predict | jq .prediction

# EHR only
curl -X POST -H "Content-Type: application/json" \
  -d '{"age":62,"gender":"Male","pack_years":35,"radon_exposure":"Low",
       "asbestos_exposure":"No","secondhand_smoke_exposure":"No",
       "copd_diagnosis":"No","alcohol_consumption":"Moderate",
       "family_history":"No"}' \
  http://127.0.0.1:5000/predict-lung-cancer

# Combined (image + EHR fields together as multipart fields)
curl -F "image=@chest.jpg" \
     -F "age=62" -F "gender=Male" -F "pack_years=35" \
     -F "radon_exposure=Low" -F "asbestos_exposure=No" \
     -F "secondhand_smoke_exposure=No" -F "copd_diagnosis=No" \
     -F "alcohol_consumption=Moderate" -F "family_history=No" \
     http://127.0.0.1:5000/predict-combined
```

---

## Web UI pages

| Route             | Template                | Purpose                                                                                                           |
| ----------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `/`               | `landing.html`          | Minimal centered hero with `Playfair Display` headline and two CTAs (**Open app**, **View evaluation**)           |
| `/dashboard`      | `dashboard.html`        | Single dominant upload card with drag & drop, an **Analyze Image** button, and a clean result panel with confidence bar |
| `/how-it-works`   | `how_it_works.html`     | Three minimal cards explaining X-ray analysis, EHR risk scoring, and combined insight                             |
| `/model-accuracy` | `accuracy.html`         | Metric spotlight cards, a fusion-score explainer, confusion matrices, performance comparison, ROC curves, and a one-click **Re-run evaluation** button |

The site uses a dark theme with warm orange accents, **Plus Jakarta Sans /
Work Sans** for body text, and **Playfair Display** for hero titles.

---

## Limitations and disclaimer

- **This is an educational / research prototype, not a medical device.**
  Output must not be used for diagnosis, triage, or treatment decisions.
- The fusion score is a heuristic blend, not a clinically validated risk
  metric. The displayed accuracy is a per-model proxy because the project
  does **not** include a paired dataset where the same patient has both an
  X-ray and an EHR record with one fused ground-truth label.
- The EHR dataset is generic / synthetic-style and does not represent any
  particular population.
- Image-model performance numbers are for the validation split with the
  per-class cap above; real-world distributions will differ.
- Grad-CAM highlights regions the model attended to; it is **interpretation
  aid only** and does not constitute a clinical finding.

---

Generated and maintained as part of the Respiratory Health AI prototype.
For the long-form design notes and history of changes, see the `docs/`
folder.
