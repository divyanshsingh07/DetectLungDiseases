"""Train a lung cancer risk classifier on the EHR dataset.

Reads `lung_cancer_dataset.csv` at the project root, builds a scikit-learn
pipeline that handles numeric and categorical features, trains a
RandomForestClassifier, and saves the fitted pipeline plus metadata.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

DATA_PATH = DATASETS_DIR / "lung_cancer_dataset.csv"
MODEL_PATH = MODELS_DIR / "lung_cancer_model.pkl"
META_PATH = MODELS_DIR / "lung_cancer_model_meta.json"

TARGET = "lung_cancer"
ID_COL = "patient_id"
NUMERIC_FEATURES = ["age", "pack_years"]
CATEGORICAL_FEATURES = [
    "gender",
    "radon_exposure",
    "asbestos_exposure",
    "secondhand_smoke_exposure",
    "copd_diagnosis",
    "alcohol_consumption",
    "family_history",
]


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    # keep_default_na=False so the literal string "None" in alcohol_consumption
    # is treated as a real category rather than a missing value.
    df = pd.read_csv(DATA_PATH, keep_default_na=False)
    required = [ID_COL, TARGET, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    # Restore numeric dtypes since keep_default_na=False reads everything as strings.
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_pipeline() -> Pipeline:
    # OneHotEncoder API changed across sklearn versions: prefer sparse_output,
    # fall back to sparse for older releases.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", ohe, CATEGORICAL_FEATURES),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def main() -> None:
    print(f"[1/5] Loading dataset from {DATA_PATH}")
    df = load_dataset()
    print(f"      Loaded {len(df):,} records | Target distribution:")
    print(df[TARGET].value_counts().to_string())

    y = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n[2/5] Building pipeline (StandardScaler + OneHotEncoder + RandomForest)")
    pipe = build_pipeline()

    print("[3/5] Training RandomForestClassifier")
    pipe.fit(X_train, y_train)

    print("[4/5] Evaluating on held-out test set")
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds).tolist()

    print(f"      Accuracy : {acc:.4f}")
    print(f"      ROC AUC  : {auc:.4f}")
    print("      Confusion Matrix [ [TN, FP], [FN, TP] ]:")
    print(f"      {cm}")
    print("\n      Classification report:")
    print(classification_report(y_test, preds, target_names=["No Cancer", "Lung Cancer"]))

    # Feature importance snapshot (after one-hot expansion)
    ohe = pipe.named_steps["preprocessor"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERIC_FEATURES + cat_names
    importances = pipe.named_steps["classifier"].feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    top_features = [
        {"feature": feature_names[i], "importance": float(importances[i])}
        for i in top_idx
    ]

    print("\n[5/5] Saving model and metadata")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)

    meta = {
        "target": TARGET,
        "target_mapping": {"No": 0, "Yes": 1},
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_value_options": {
            col: sorted(df[col].dropna().unique().tolist())
            for col in CATEGORICAL_FEATURES
        },
        "metrics": {
            "accuracy": round(float(acc), 4),
            "roc_auc": round(float(auc), 4),
            "confusion_matrix": cm,
        },
        "top_features": top_features,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"      Model  -> {MODEL_PATH}")
    print(f"      Meta   -> {META_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
