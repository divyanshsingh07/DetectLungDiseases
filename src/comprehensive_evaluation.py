"""
Comprehensive Model Evaluation Script.

Evaluates three models that ship with the project:
  1. Multi-Disease Chest X-ray Classifier  (5 classes, ResNet18)
  2. Pneumonia Detection                    (legacy 2 classes, ResNet18)
  3. Lung Cancer Risk                       (sklearn pipeline on EHR data)

Outputs into ``evaluation_results/``:
  - confusion_matrices.png
  - metrics_comparison.png
  - roc_curves.png
  - results_table.csv
  - evaluation_report.json (includes ``combined_metrics.fusion_proxy``: multimodal
    accuracy proxy when both X-ray and EHR ran, else X-ray-only or EHR-only)
"""

from __future__ import annotations

import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as torchvision_models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets"
EVAL_RESULTS_DIR = BASE_DIR / "evaluation_results"
EVAL_RESULTS_DIR.mkdir(exist_ok=True)

# Score-level fusion weights (keep in sync with app.FUSION_WEIGHTS).
FUSION_WEIGHTS = {"xray_abnormal": 0.5, "ehr_lung_cancer": 0.5}

sys.path.insert(0, str(SRC_DIR))

from data_loader import get_dataloaders  # noqa: E402
from model import get_model  # noqa: E402
from train_multi_disease import (  # noqa: E402
    CLASS_NAMES as MULTI_CLASS_NAMES,
    CLASS_SOURCES as MULTI_CLASS_SOURCES,
    ChestXrayDataset,
    build_transforms,
    collect_samples,
    validate_class_counts,
)


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (15, 10)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ComprehensiveModelEvaluator:
    """Evaluates all medical prediction models and writes a unified report."""

    def __init__(self):
        self.results: Dict[str, dict] = {}
        self.device = pick_device()
        print(f"Using device: {self.device}")

    # ------------------------------------------------------------------
    # Multi-disease X-ray model
    # ------------------------------------------------------------------
    def evaluate_multi_disease_model(self, max_per_class: int = 1500, seed: int = 42):
        print("\n" + "=" * 70)
        print("EVALUATING MULTI-DISEASE CHEST X-RAY MODEL")
        print("=" * 70)
        model_path = MODELS_DIR / "multi_disease_model.pth"
        meta_path = MODELS_DIR / "multi_disease_meta.json"
        if not model_path.exists():
            print("Multi-disease model artefact missing - skipping.")
            return None
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        class_names: List[str] = meta.get("class_names", MULTI_CLASS_NAMES)

        print(f"Classes: {class_names}")

        try:
            paths, labels, counts = collect_samples(max_per_class=max_per_class, seed=seed)
            validate_class_counts(counts)
            _, paths_val, _, y_val = train_test_split(
                paths, labels, test_size=0.2, random_state=seed, stratify=labels
            )
            print(f"Validation samples: {len(paths_val)}")
        except Exception as exc:
            print(f"Failed to rebuild validation set: {exc}")
            return None

        _, eval_tf = build_transforms()
        ds = ChestXrayDataset(paths_val, y_val, eval_tf)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

        net = torchvision_models.resnet18(weights=None)
        net.fc = nn.Linear(net.fc.in_features, len(class_names))
        try:
            state = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location=self.device)
        net.load_state_dict(state)
        net.to(self.device)
        net.eval()

        all_preds: List[int] = []
        all_labels: List[int] = []
        all_probs: List[List[float]] = []
        with torch.no_grad():
            for i, (imgs, lbls) in enumerate(loader):
                if i % 5 == 0:
                    print(f"  batch {i}/{len(loader)}")
                imgs = imgs.to(self.device)
                logits = net(imgs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_preds.extend(preds.tolist())
                all_labels.extend(lbls.tolist())
                all_probs.extend(probs.tolist())

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

        # Treat any non-Normal class as the positive label for ROC purposes
        normal_idx = class_names.index("Normal") if "Normal" in class_names else 0
        binary_labels = [0 if l == normal_idx else 1 for l in all_labels]
        abnormal_probs = [
            sum(p for j, p in enumerate(probs) if j != normal_idx)
            for probs in all_probs
        ]

        self.results["multi_disease"] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "confusion_matrix": cm,
            "predictions": np.array(all_preds),
            "labels": np.array(all_labels),
            "probabilities": np.array(abnormal_probs),
            "binary_labels": np.array(binary_labels),
            "class_names": class_names,
            "num_samples": len(all_labels),
            "per_class_counts": counts,
        }
        print(f"\n[multi_disease] acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
        return self.results["multi_disease"]

    # ------------------------------------------------------------------
    # Legacy 2-class pneumonia model
    # ------------------------------------------------------------------
    def evaluate_pneumonia_model(self):
        print("\n" + "=" * 70)
        print("EVALUATING LEGACY PNEUMONIA DETECTION MODEL")
        print("=" * 70)
        model_path = MODELS_DIR / "model.pth"
        if not model_path.exists():
            print("Legacy model.pth missing - skipping.")
            return None
        try:
            print("Loading model...")
            model = get_model(num_classes=2)
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=True)
            except TypeError:
                state = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()

            print("Loading validation data...")
            _, val_loader = get_dataloaders("chest_xray", batch_size=32)
            all_preds, all_labels, all_probs = [], [], []
            with torch.no_grad():
                for i, (images, labels) in enumerate(val_loader):
                    if i % 10 == 0:
                        print(f"  Batch {i}/{len(val_loader)}")
                    images = images.to(self.device)
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            prec = precision_score(all_labels, all_preds, zero_division=0)
            rec = recall_score(all_labels, all_preds, zero_division=0)
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            cm = confusion_matrix(all_labels, all_preds)

            self.results["pneumonia"] = {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "confusion_matrix": cm,
                "predictions": np.array(all_preds),
                "labels": np.array(all_labels),
                "probabilities": np.array(all_probs),
                "class_names": ["Normal", "Pneumonia"],
                "num_samples": len(all_labels),
            }
            print(f"\n[pneumonia] acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
            return self.results["pneumonia"]
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"Error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Lung cancer EHR model
    # ------------------------------------------------------------------
    def evaluate_lung_cancer_model(self):
        print("\n" + "=" * 70)
        print("EVALUATING LUNG CANCER RISK MODEL")
        print("=" * 70)
        try:
            with open(MODELS_DIR / "lung_cancer_model.pkl", "rb") as f:
                model = pickle.load(f)
            with open(MODELS_DIR / "lung_cancer_model_meta.json") as f:
                meta = json.load(f)

            df = pd.read_csv(DATASETS_DIR / "lung_cancer_dataset.csv", keep_default_na=False)
            y = df["lung_cancer"].map({"Yes": 1, "No": 0}).astype(int)
            X = df[meta["numeric_features"] + meta["categorical_features"]]
            for col in meta["numeric_features"]:
                X[col] = pd.to_numeric(X[col], errors="coerce")

            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            cm = confusion_matrix(y_test, preds)

            self.results["lung_cancer"] = {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "confusion_matrix": cm,
                "predictions": preds,
                "labels": y_test.values,
                "probabilities": probs,
                "class_names": ["No Cancer", "Lung Cancer"],
                "num_samples": len(y_test),
            }
            print(f"\n[lung_cancer] acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
            return self.results["lung_cancer"]
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"Error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Combined / aggregate metrics
    # ------------------------------------------------------------------
    def calculate_combined_metrics(self):
        print("\n" + "=" * 70)
        print("CALCULATING COMBINED METRICS")
        print("=" * 70)
        keys = [k for k in ("multi_disease", "pneumonia", "lung_cancer") if k in self.results]
        if not keys:
            print("No model results - cannot combine.")
            return None
        avg_acc = np.mean([self.results[k]["accuracy"] for k in keys])
        avg_prec = np.mean([self.results[k]["precision"] for k in keys])
        avg_rec = np.mean([self.results[k]["recall"] for k in keys])
        avg_f1 = np.mean([self.results[k]["f1"] for k in keys])
        total_samples = sum(self.results[k]["num_samples"] for k in keys)
        weighted_acc = sum(
            self.results[k]["accuracy"] * self.results[k]["num_samples"] for k in keys
        ) / max(1, total_samples)

        self.results["combined"] = {
            "average_accuracy": float(avg_acc),
            "average_precision": float(avg_prec),
            "average_recall": float(avg_rec),
            "average_f1": float(avg_f1),
            "weighted_accuracy": float(weighted_acc),
            "total_samples": int(total_samples),
            "models_evaluated": keys,
        }
        self._attach_fusion_proxy()
        print(
            f"avg_acc={avg_acc:.4f} weighted_acc={weighted_acc:.4f} "
            f"models={keys} samples={total_samples}"
        )
        return self.results["combined"]

    def _attach_fusion_proxy(self):
        """App-style accuracy proxy: both modalities when available, else X-ray or EHR only."""
        c = self.results.get("combined")
        if not c:
            return
        has_ehr = "lung_cancer" in self.results
        xray_key = None
        if "multi_disease" in self.results:
            xray_key = "multi_disease"
        elif "pneumonia" in self.results:
            xray_key = "pneumonia"

        wx = float(FUSION_WEIGHTS["xray_abnormal"])
        we = float(FUSION_WEIGHTS["ehr_lung_cancer"])
        wt = wx + we
        if wt <= 0:
            return

        if xray_key and has_ehr:
            x_acc = self.results[xray_key]["accuracy"]
            e_acc = self.results["lung_cancer"]["accuracy"]
            proxy = (wx / wt) * x_acc + (we / wt) * e_acc
            c["fusion_proxy"] = {
                "mode": "multimodal",
                "accuracy_proxy": float(proxy),
                "xray_model": xray_key,
                "xray_weight": wx / wt,
                "ehr_weight": we / wt,
            }
        elif xray_key:
            c["fusion_proxy"] = {
                "mode": "xray_only",
                "accuracy_proxy": float(self.results[xray_key]["accuracy"]),
                "xray_model": xray_key,
                "xray_weight": 1.0,
                "ehr_weight": 0.0,
            }
        elif has_ehr:
            c["fusion_proxy"] = {
                "mode": "ehr_only",
                "accuracy_proxy": float(self.results["lung_cancer"]["accuracy"]),
                "xray_model": None,
                "xray_weight": 0.0,
                "ehr_weight": 1.0,
            }

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def plot_confusion_matrices(self, save_path=None):
        if save_path is None:
            save_path = str(EVAL_RESULTS_DIR / "confusion_matrices.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        models_to_plot = [k for k in ("multi_disease", "pneumonia", "lung_cancer") if k in self.results]
        n = len(models_to_plot)
        if n == 0:
            return
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]
        cmaps = {"multi_disease": "Purples", "pneumonia": "Blues", "lung_cancer": "Greens"}
        titles = {
            "multi_disease": "Multi-Disease X-ray",
            "pneumonia": "Pneumonia (legacy)",
            "lung_cancer": "Lung Cancer Risk (EHR)",
        }
        for ax, key in zip(axes, models_to_plot):
            r = self.results[key]
            cm = r["confusion_matrix"]
            cls = r["class_names"]
            sns.heatmap(cm, annot=True, fmt="d", cmap=cmaps[key], xticklabels=cls,
                        yticklabels=cls, ax=ax, cbar_kws={"label": "Count"})
            ax.set_title(f'{titles[key]}\nAccuracy: {r["accuracy"]:.2%}', fontsize=11, fontweight="bold")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    def plot_metrics_comparison(self, save_path=None):
        if save_path is None:
            save_path = str(EVAL_RESULTS_DIR / "metrics_comparison.png")
        metrics = ["accuracy", "precision", "recall", "f1"]
        labels = ["Accuracy", "Precision", "Recall", "F1"]
        models_to_plot = [k for k in ("multi_disease", "pneumonia", "lung_cancer") if k in self.results]
        if not models_to_plot:
            return
        x = np.arange(len(metrics))
        width = 0.8 / max(1, len(models_to_plot))
        colors = {"multi_disease": "mediumpurple", "pneumonia": "steelblue", "lung_cancer": "seagreen"}
        names = {"multi_disease": "Multi-Disease X-ray", "pneumonia": "Pneumonia (legacy)", "lung_cancer": "Lung Cancer (EHR)"}
        fig, ax = plt.subplots(figsize=(11, 6))
        for i, key in enumerate(models_to_plot):
            scores = [self.results[key][m] for m in metrics]
            offset = (i - (len(models_to_plot) - 1) / 2) * width
            bars = ax.bar(x + offset, scores, width, label=names[key], color=colors[key], alpha=0.85)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    def plot_roc_curves(self, save_path=None):
        if save_path is None:
            save_path = str(EVAL_RESULTS_DIR / "roc_curves.png")
        models_to_plot = []
        if "multi_disease" in self.results:
            r = self.results["multi_disease"]
            models_to_plot.append(("Multi-Disease (Normal vs Abnormal)", r["binary_labels"], r["probabilities"], "purple"))
        if "pneumonia" in self.results:
            r = self.results["pneumonia"]
            models_to_plot.append(("Pneumonia (legacy)", r["labels"], r["probabilities"], "darkorange"))
        if "lung_cancer" in self.results:
            r = self.results["lung_cancer"]
            models_to_plot.append(("Lung Cancer Risk", r["labels"], r["probabilities"], "green"))
        if not models_to_plot:
            return
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, lbls, probs, color in models_to_plot:
            fpr, tpr, _ = roc_curve(lbls, probs)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.3f})", color=color)
        ax.plot([0, 1], [0, 1], "--", color="navy", lw=1.5, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------
    def create_results_table(self, save_path=None):
        if save_path is None:
            save_path = str(EVAL_RESULTS_DIR / "results_table.csv")
        rows = []
        order = [
            ("multi_disease", "Multi-Disease X-ray"),
            ("pneumonia", "Pneumonia (legacy)"),
            ("lung_cancer", "Lung Cancer Risk"),
        ]
        for key, name in order:
            if key in self.results:
                r = self.results[key]
                rows.append({
                    "Model": name,
                    "Accuracy": f"{r['accuracy']:.4f}",
                    "Precision": f"{r['precision']:.4f}",
                    "Recall": f"{r['recall']:.4f}",
                    "F1-Score": f"{r['f1']:.4f}",
                    "Samples": r["num_samples"],
                })
        if "combined" in self.results:
            c = self.results["combined"]
            rows.append({
                "Model": "Combined (Average)",
                "Accuracy": f"{c['average_accuracy']:.4f}",
                "Precision": f"{c['average_precision']:.4f}",
                "Recall": f"{c['average_recall']:.4f}",
                "F1-Score": f"{c['average_f1']:.4f}",
                "Samples": c["total_samples"],
            })
            rows.append({
                "Model": "Combined (Weighted)",
                "Accuracy": f"{c['weighted_accuracy']:.4f}",
                "Precision": "-", "Recall": "-", "F1-Score": "-",
                "Samples": c["total_samples"],
            })
            fp = c.get("fusion_proxy")
            if fp:
                rows.append({
                    "Model": f"Fusion proxy ({fp['mode']})",
                    "Accuracy": f"{fp['accuracy_proxy']:.4f}",
                    "Precision": "-", "Recall": "-", "F1-Score": "-",
                    "Samples": "-",
                })
        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        print(f"  Saved: {save_path}")
        return df

    def save_json_report(self, save_path=None):
        if save_path is None:
            save_path = str(EVAL_RESULTS_DIR / "evaluation_report.json")
        def serialise(key: str) -> dict:
            r = self.results[key]
            return {
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1_score": r["f1"],
                "num_samples": r["num_samples"],
                "confusion_matrix": np.asarray(r["confusion_matrix"]).tolist(),
                "class_names": r["class_names"],
            }
        report = {}
        if "multi_disease" in self.results:
            report["multi_disease_model"] = serialise("multi_disease")
            report["multi_disease_model"]["per_class_counts"] = self.results["multi_disease"].get("per_class_counts", {})
        if "pneumonia" in self.results:
            report["pneumonia_model"] = serialise("pneumonia")
        if "lung_cancer" in self.results:
            report["lung_cancer_model"] = serialise("lung_cancer")
        if "combined" in self.results:
            report["combined_metrics"] = self.results["combined"]
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Saved: {save_path}")

    def print_classification_reports(self):
        print("\n" + "=" * 70)
        print("DETAILED CLASSIFICATION REPORTS")
        print("=" * 70)
        for key, name in (
            ("multi_disease", "Multi-Disease X-ray"),
            ("pneumonia", "Pneumonia (legacy)"),
            ("lung_cancer", "Lung Cancer Risk"),
        ):
            if key not in self.results:
                continue
            r = self.results[key]
            print(f"\n{name}:")
            print("-" * 70)
            print(classification_report(r["labels"], r["predictions"],
                                        target_names=r["class_names"], digits=4))

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------
    def generate_full_report(self):
        print("\n" + "=" * 70)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("=" * 70)

        self.evaluate_multi_disease_model()
        self.evaluate_pneumonia_model()
        self.evaluate_lung_cancer_model()
        if not self.results:
            print("\nNo models could be evaluated.")
            return None

        self.calculate_combined_metrics()
        print("\nGenerating visualisations...")
        self.plot_confusion_matrices()
        self.plot_metrics_comparison()
        self.plot_roc_curves()
        print("\nWriting reports...")
        results_df = self.create_results_table()
        self.save_json_report()
        self.print_classification_reports()
        print("\nFINAL SUMMARY")
        print(results_df.to_string(index=False))
        print("\nEvaluation complete. See evaluation_results/.\n")
        return self.results


def main():
    evaluator = ComprehensiveModelEvaluator()
    results = evaluator.generate_full_report()
    if results is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
