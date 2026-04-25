# How training works

This project trains **three separate models** used at inference time in `app.py`. Each trainer is a standalone script under **`src/`**; they read data from **`datasets/`** and write artefacts into **`models/`**.

There is **no single training script** that trains “fusion” — fusion is a **weighted average** of two already-trained models, applied only in the Flask app.

---

## Prerequisites

1. **Python environment** — same stack as `requirements.txt` (PyTorch, torchvision, scikit-learn, pandas, Pillow, etc.).
2. **Working directory** — run commands from the **project root** (`pneumonia-detection/`), e.g.  
   `python src/train_multi_disease.py`
3. **Data on disk** — each trainer expects specific paths under `datasets/` (see each section below). If folders or the CSV are missing, the script will error or warn and may skip empty classes.

---

## Overview: the three trainers

| Script | Task | Framework | Output files |
| ------ | ---- | --------- | ------------ |
| `src/train_multi_disease.py` | 5-class chest X-ray (Normal, Pneumonia, COVID-19, Tuberculosis, Lung_Opacity) | PyTorch ResNet18 | `models/multi_disease_model.pth`, `models/multi_disease_meta.json` |
| `src/train.py` | Legacy 2-class (Normal vs Pneumonia) | PyTorch ResNet18 | `models/model.pth` |
| `src/train_lung_cancer.py` | Lung cancer Yes/No from EHR tabular features | scikit-learn pipeline | `models/lung_cancer_model.pkl`, `models/lung_cancer_model_meta.json` |

---

## 1. Multi-disease X-ray training (`train_multi_disease.py`)

### What it does

1. **Collects** image paths from several folders under `datasets/`, mapped to five fixed class names (`CLASS_NAMES` in code). It walks each directory recursively, keeps only `.png` / `.jpg` / `.jpeg`, and **skips** COVID dataset **`masks`** subfolders.
2. **Caps** each class at `--max-per-class` (default **1500**) after a deterministic shuffle, so huge “Normal” pools do not dominate training.
3. **Validates** that every class has at least **2** images (needed for stratified train/validation split).
4. **Splits** with `sklearn.model_selection.train_test_split`: **80% train / 20% val**, `stratify=labels`, fixed `random_state`.
5. **Dataset class** `ChestXrayDataset`: opens each path with PIL, converts to RGB; on read failure uses a black 224×224 placeholder so one corrupt file does not stop the epoch.
6. **Transforms**
   - **Train:** resize 224×224, random horizontal flip, random rotation ±10°, light color jitter, to tensor, ImageNet normalize.
   - **Val:** resize, to tensor, normalize (no random aug).
7. **Model:** `torchvision.models.resnet18` with **ImageNet pretrained** weights. All parameters frozen except **`layer4`** and the new **`fc`** head (`nn.Linear` → 5 logits).
8. **Loss:** `CrossEntropyLoss` with **per-class inverse-frequency weights** computed on the **training** label counts (helps rarer classes).
9. **Optimizer:** Adam on **trainable** parameters only, default **lr = 1e-3**.
10. **Loop:** For each epoch, train pass then validation pass (`run_epoch`). Tracks loss and accuracy; whenever **validation accuracy** improves, saves `state_dict` to `multi_disease_model.pth`.
11. **After training:** Reloads the **best** checkpoint from disk, re-runs validation once, prints classification report, then writes **`multi_disease_meta.json`** (class names, source map, counts, hyperparameters, per-epoch history, final metrics, confusion matrix).

### CLI flags

| Flag | Default | Role |
| ---- | ------- | ---- |
| `--epochs` | 3 | Number of full passes over the training set |
| `--batch-size` | 32 | Mini-batch size |
| `--max-per-class` | 1500 | Max images sampled per disease class |
| `--lr` | 0.001 | Adam learning rate |
| `--workers` | 2 | `DataLoader` worker processes |
| `--seed` | 42 | Python / NumPy / PyTorch RNG + split reproducibility |

### Device

Uses **CUDA** if available, else **Apple MPS** if available, else **CPU** (`pick_device()`).

### Command

```bash
python src/train_multi_disease.py
python src/train_multi_disease.py --epochs 5 --max-per-class 2000 --batch-size 32
```

---

## 2. Legacy pneumonia training (`train.py`)

### What it does

1. **`data_loader.get_dataloaders("chest_xray")`** resolves to `datasets/chest_xray/train/` and uses `torchvision.datasets.ImageFolder` (subfolder names = class names, typically NORMAL / PNEUMONIA).
2. **Split:** 80% train / 20% validation via `random_split` on the full train folder (not a separate Kaggle `val/` folder).
3. **Model:** `model.get_model(num_classes=2)` — ResNet18 pretrained, **entire backbone frozen**, only **`fc`** replaced with a 2-class linear layer and **trained**.
4. **Loss / optimizer:** `CrossEntropyLoss`, Adam with **lr=0.001** on **`model.fc.parameters()`** only.
5. **Epochs:** **`num_epochs = 1`** in the script (quick default); each epoch runs one train loop and one val loop, prints loss/accuracy, saves **`models/model.pth`** when validation accuracy improves.

### Command

```bash
python src/train.py
```

### Note

The Flask app prefers **`multi_disease_model.pth`**. If that file is missing, inference falls back to this **2-class** checkpoint (`model.pth`) and a simpler abnormal probability (`1 − P(Normal)`).

---

## 3. Lung cancer EHR training (`train_lung_cancer.py`)

### What it does

1. **Loads** `datasets/lung_cancer_dataset.csv` with `keep_default_na=False` so the literal category string `"None"` in `alcohol_consumption` is not parsed as NaN.
2. **Columns:** requires `patient_id`, `lung_cancer`, numeric `age` / `pack_years`, and seven categorical fields (gender, exposures, COPD, alcohol, family history).
3. **Target:** maps `lung_cancer` **Yes → 1**, **No → 0**.
4. **Split:** `train_test_split` with **test_size=0.2**, **stratify=y**, `random_state=42`.
5. **Pipeline (`build_pipeline`):**
   - **`ColumnTransformer`:** `StandardScaler` on numeric columns; **`OneHotEncoder(handle_unknown="ignore")`** on categoricals (with a fallback for older sklearn `sparse=` API).
   - **`RandomForestClassifier`:** 300 trees, `min_samples_leaf=2`, `class_weight="balanced"`, `n_jobs=-1`, `random_state=42`.
6. **`pipe.fit(X_train, y_train)`** — single fit; no separate “epoch” loop (tree ensemble is not iterative like SGD).
7. **Evaluation** on the held-out test set: accuracy, ROC-AUC, confusion matrix, classification report.
8. **Feature importances** — reads the fitted OHE feature names, concatenates with numeric names, sorts importances, keeps top 10 for metadata.
9. **Saves** `lung_cancer_model.pkl` (full fitted pipeline via `pickle`) and `lung_cancer_model_meta.json` (feature lists, allowed categorical values from the CSV, metrics, train/test sizes, top features).

### Command

```bash
python src/train_lung_cancer.py
```

---

## What is *not* trained here

- **Fusion** (`0.5 × abnormal_xray + 0.5 × lung_cancer_probability`) is **not** learned from data in this repo; it is **fixed logic** in `app.py` (`fuse_patient_risk`). You can change weights there without retraining.
- **`comprehensive_evaluation.py`** only **evaluates** existing checkpoints; it does not update weights.

---

## After training: how artefacts are used

| File | Consumer |
| ---- | -------- |
| `models/multi_disease_model.pth` + `multi_disease_meta.json` | `app.py` — primary X-ray inference + class order |
| `models/model.pth` | `app.py` — fallback 2-class X-ray if multi checkpoint absent |
| `models/lung_cancer_model.pkl` + `lung_cancer_model_meta.json` | `app.py` — EHR validation schema + `predict_proba` |

To refresh charts and JSON metrics after retraining, run:

```bash
python src/comprehensive_evaluation.py
```

or use **Re-run evaluation** on `/model-accuracy`.

---

## Quick reference commands

```bash
# All three (order only matters if you want X-ray trainers to finish first)
python src/train_multi_disease.py --epochs 3 --max-per-class 1500
python src/train.py
python src/train_lung_cancer.py

# Then regenerate evaluation assets (optional)
python src/comprehensive_evaluation.py
```

This is the full **training** side of the pipeline; for how requests flow at inference time, see **`pipline.md`**.
