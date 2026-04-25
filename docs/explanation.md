# Complete project explanation

This document explains **how the whole Respiratory Health AI project fits together**: data, models, training, the web app, **every score the user sees**, how evaluation numbers are produced, and how they should be interpreted.

For deeper detail on specific areas:

- **End-to-end data and request flow** → `pipline.md`
- **Training scripts and hyperparameters** → `training.md`
- **Install, run commands, repo layout** → `README.md`

---

## 1. What this project is

**Respiratory Health AI** is a research / educational prototype that combines:

1. **Chest X-ray analysis** — a deep neural network classifies the image into lung-related categories (multi-disease) or, if that model is missing, a simpler **Normal vs Pneumonia** model.
2. **Electronic health record (EHR) analysis** — a classical machine learning pipeline estimates **lung cancer risk** from tabular fields (age, smoking, exposures, etc.).
3. **Fusion** — when both an image and EHR fields are submitted together, the app computes a **single combined score** by blending two probabilities with fixed weights (default: **50% / 50%**).

This is **multi-model, late fusion**: two separate models, two modalities, merged **only at the output** in Python. It is **not** one neural network that jointly learns image + tabular inputs (that would need a paired multimodal dataset with one label per patient for both inputs).

---

## 2. High-level architecture

```text
┌─────────────────┐     ┌──────────────────────────┐
│  User / API     │     │  Flask (`app.py`)        │
│  image +/or EHR │────▶│  load models at startup  │
└─────────────────┘     └────────────┬─────────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
   ┌───────────────┐         ┌───────────────┐         ┌─────────────────┐
   │ ResNet18      │         │ ResNet18      │         │ sklearn         │
   │ multi-disease │   OR    │ legacy 2-class│         │ Pipeline        │
   │ (.pth)        │         │ (.pth)        │         │ RF + preprocess │
   └───────┬───────┘         └───────┬───────┘         └────────┬────────┘
           │                         │                          │
           │  softmax probs          │  softmax probs           │  P(cancer)
           └────────────┬────────────┴──────────────┬───────────┘
                        │                           │
                        ▼                           ▼
                 abnormal %                  cancer risk %
                        └───────────┬───────────┘
                                    ▼
                          fuse_patient_risk()
                          (weighted average)
                                    ▼
                          combined % + risk band
```

**Artifacts** live in `models/`; **raw data** under `datasets/`; **evaluation plots and JSON** under `evaluation_results/`; **UI** in `templates/` + `static/`.

---

## 3. Models and what each one outputs

### 3.1 Multi-disease chest X-ray model (primary)

- **File:** `models/multi_disease_model.pth` (+ `multi_disease_meta.json`)
- **Architecture:** ResNet18, ImageNet-pretrained; typically last residual block + classifier trained on your chest data.
- **Classes (5):** Normal, Pneumonia, COVID-19, Tuberculosis, Lung_Opacity.
- **Training:** `src/train_multi_disease.py` — images merged from three public datasets under `datasets/`, balanced with a per-class cap.

**At inference**, for one image the app computes:

| Field | Meaning |
| ----- | ------- |
| **`prediction`** | Argmax class name (the model’s single best label). |
| **`confidence`** | Softmax probability of that top class, as **0–100%**. |
| **`probabilities`** | One percentage per class (softmax for each class). They sum to **100%** (up to rounding). |
| **`abnormal_probability`** | Sum of softmax mass over every class **except Normal** (Pneumonia + COVID-19 + Tuberculosis + Lung_Opacity). This is the scalar used in **fusion** as the “X-ray side” of the story. |
| **`heatmap_overlay`** | Grad-CAM on the last conv block, blended onto the image — shows **where** the model focused; not a clinical diagnosis. |

If **`multi_disease_model.pth` is missing**, the app loads the legacy 2-class model instead: **`abnormal_probability` ≈ 1 − P(Normal)**.

### 3.2 Legacy 2-class pneumonia model (fallback)

- **File:** `models/model.pth`
- **Classes:** Normal, Pneumonia.
- **Training:** `src/train.py` + `src/data_loader.py` on `datasets/chest_xray/train/`.

Used only when the multi-disease checkpoint is unavailable; same API shape for X-ray JSON, but only two probabilities.

### 3.3 Lung cancer risk from EHR

- **Files:** `models/lung_cancer_model.pkl`, `models/lung_cancer_model_meta.json`
- **Pipeline:** Scale numeric features → one-hot categoricals → **RandomForest** → `predict_proba`.
- **Training:** `src/train_lung_cancer.py` on `datasets/lung_cancer_dataset.csv`.

**At inference:**

| Field | Meaning |
| ----- | ------- |
| **`probability`** | Model’s estimated **P(lung cancer \| features)**, as **0–100%** (positive class probability). |
| **`risk_band`** | Human-readable tier (e.g. Low / Moderate / High / Very High) derived from thresholds on that probability. |
| **Recommendations** | Rule-based text hints from `app.py` (not from the forest itself). |

---

## 4. Fusion: how the “combined” score works

**Endpoint:** `POST /predict-combined` (multipart: file `image` + EHR form fields).

**Steps:**

1. Run X-ray inference → get **`abnormal_probability`** (0–100).
2. Run EHR pipeline → get **`probability`** (0–100).
3. Convert both to **0–1**, clip to `[0, 1]`.
4. Apply **normalized weights** from `FUSION_WEIGHTS` in `app.py` (default each **0.5**):

   ```text
   combined_01 = w_xray × (abnormal/100) + w_ehr × (cancer/100)
   combined_score = combined_01 × 100
   ```

5. Map **`combined_01`** to a **combined risk band** (same style of tiers as the EHR-only band, for consistent UI language).

**Important:**

- The combined number is a **heuristic blend** for demonstration. It is **not** trained end-to-end and **not** validated against a real “fused disease” label, because this repo does not ship a dataset where **the same patient** has paired X-ray + EHR + one fusion ground-truth outcome.
- Changing fusion behaviour = edit weights (or logic) in **`app.py`**; no retraining required.

---

## 5. Web application: how the user experiences it

| URL | Role |
| --- | ---- |
| **`/`** | Landing — product intro and links. |
| **`/dashboard`** | Main flow: upload X-ray → `POST /predict` → shows prediction, confidence bar, class probabilities, recommendations. (Grad-CAM is returned by the API; the minimal UI may not always show every analysis widget.) |
| **`/how-it-works`** | Short explanation of X-ray, EHR, and fusion. |
| **`/model-accuracy`** | Reads `evaluation_results/evaluation_report.json`, shows **accuracy / precision / recall / F1**, confusion-matrix images, ROC, metric comparison chart, and a button to **re-run** `src/comprehensive_evaluation.py`. |

**Static assets** (`CSS`, `JS`) are served from `/static/...`. The browser talks only to Flask; there is no separate Node backend.

---

## 6. Scores on the metrics page (`/model-accuracy`)

These numbers come from **`src/comprehensive_evaluation.py`**, which loads the **same** checkpoints and evaluates them on held-out or validation-style splits, then writes **`evaluation_results/evaluation_report.json`** and **`results_table.csv`**.

### 6.1 Per-model metrics (standard definitions)

Let **positive** mean “the class we care about” in context; for multi-class, macro averages treat each class symmetrically.

| Metric | Plain-language meaning |
| ------ | ------------------------ |
| **Accuracy** | Fraction of samples where the **predicted class** equals the **true class**. |
| **Precision** (macro for multi-class) | Averaged per-class: of all predictions for class *k*, how many were correct. High precision → fewer false alarms for that class. |
| **Recall** (macro) | Averaged per-class: of all **true** class-*k* samples, how many the model found. High recall → fewer misses for that class. |
| **F1** (macro) | Harmonic mean of precision and recall per class, then averaged — a single balance when classes matter equally in the average. |

**Example snapshot** (from `evaluation_results/results_table.csv` — your run may differ slightly after re-evaluation):

| Model | Accuracy | Precision | Recall | F1 | Samples |
| ----- | -------- | --------- | ------ | --- | ------- |
| Multi-Disease X-ray | 0.9239 | 0.9338 | 0.9297 | 0.9308 | 1340 |
| Pneumonia (legacy) | 0.9272 | 0.9390 | 0.9654 | 0.9520 | 1044 |
| Lung Cancer Risk | 0.6982 | 0.7793 | 0.7825 | 0.7809 | 10000 |

- **X-ray rows** use the **validation split** size shown (e.g. 1340 for multi-disease after the train/val split used in that script).
- **Lung cancer row** uses the **test split** of the CSV pipeline (e.g. 10000 test rows in the default split).

### 6.2 “Combined” rows in the table

These are **not** a fourth trained model’s test accuracy:

- **Combined (Average)** — arithmetic mean of the three models’ accuracy, precision, recall, and F1. It is a **summary statistic** for the dashboard only.
- **Combined (Weighted)** — average accuracy **weighted by sample count** per model (so the EHR model’s 10k samples pull the average more than the X-ray val sets). Precision/recall/F1 are not shown for that row in the CSV because a single weighted scalar across heterogeneous tasks is not standardised the same way.

**There is no confusion matrix or ROC for “fusion” as a classifier** in the same sense as the three models, because fusion does not have its own set of paired labels. The UI explains this: fusion is a **post-hoc score**, not a jointly supervised classifier.

### 6.3 Confusion matrices

Each square shows **true label (rows) vs predicted label (columns)**. Diagonal = correct; off-diagonal = confusions between diseases (or No Cancer vs Lung Cancer for EHR).

### 6.4 ROC curves

For **binary** or **one-vs-rest** views, the evaluator plots how true-positive rate trades off against false-positive rate as the threshold moves. Multi-class X-ray curves are derived in the evaluation script’s logic (see `comprehensive_evaluation.py` for the exact binarisation used per plot).

---

## 7. API responses vs dashboard numbers

| Context | Where scores appear |
| ------- | ------------------- |
| Live upload on `/dashboard` | JSON from `/predict`: `prediction`, `confidence`, `probabilities`, `abnormal_probability`, optional analysis. |
| EHR form (if you add or call it) | `/predict-lung-cancer`: `probability`, `risk_band`. |
| Both together | `/predict-combined`: nested `xray`, `lung_cancer`, `combined` with `score`, `weights`, `components`, `summary`. |
| Historical / batch metrics | `/model-accuracy` + files in `evaluation_results/`. |

Live inference uses **the same mathematical definitions** (softmax, `predict_proba`) as during evaluation; the **numeric values** on a new image will differ from the aggregate metrics on the metrics page.

---

## 8. End-to-end lifecycle (one paragraph)

**Datasets** are placed under `datasets/`. **Training scripts** in `src/` fit models and save **`models/*.pth`** and **`models/*.pkl`** plus JSON metadata. **`app.py`** loads those once at process start. Each HTTP request runs the relevant forward pass(s), optionally **fuses** two scalar risks, and returns JSON. **`comprehensive_evaluation.py`** recomputes aggregate **accuracy, precision, recall, F1, confusion matrices, and ROC** into **`evaluation_results/`**, which the **model-accuracy** page displays. The **combined** table rows summarise the three models; they do **not** represent a separately trained fusion classifier.

---

## 9. Limitations (read before citing “scores”)

- **Not a medical device** — outputs are for learning and demonstration, not diagnosis or treatment.
- **Datasets** are public or generic; real clinics have different scanners, populations, and prevalence.
- **Fusion** is a weighted average of two different risk constructs (X-ray abnormality spread vs cancer probability); interpret it as a **demo index**, not a validated clinical score.
- **Grad-CAM** highlights model attention, not pathology confirmed by a radiologist.

---

## 10. File map (mental model)

| Concern | Primary location |
| ------- | ---------------- |
| Inference, fusion, routes | `app.py` |
| Train multi-disease X-ray | `src/train_multi_disease.py` |
| Train legacy X-ray | `src/train.py`, `src/data_loader.py`, `src/model.py` |
| Train EHR | `src/train_lung_cancer.py` |
| Batch metrics & plots | `src/comprehensive_evaluation.py` → `evaluation_results/` |
| UI | `templates/`, `static/` |

You now have a single narrative from **raw data → trained files → live scores → dashboard metrics**, with an explicit distinction between **per-request probabilities** and **offline evaluation aggregates**.
