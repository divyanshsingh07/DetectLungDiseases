"""Train a unified chest X-ray multi-disease classifier.

Fuses three public datasets sitting in the project root into a single
5-class ResNet18 classifier:

  Class mapping (target -> source folders):
    Normal          <- chest_xray/train/NORMAL
                       COVID-19_Radiography_Dataset/Normal/images
                       TB_Chest_Radiography_Database/Normal
    Pneumonia       <- chest_xray/train/PNEUMONIA
                       COVID-19_Radiography_Dataset/Viral Pneumonia/images
    COVID-19        <- COVID-19_Radiography_Dataset/COVID/images
    Tuberculosis    <- TB_Chest_Radiography_Database/Tuberculosis
    Lung_Opacity    <- COVID-19_Radiography_Dataset/Lung_Opacity/images

The script caps the per-class sample count so the training set stays
balanced (the "Normal" folders collectively have ~15K images which
otherwise dominate the gradient). The output artefacts are:

    multi_disease_model.pth   -> ResNet18 state_dict (5-class head)
    multi_disease_meta.json   -> class names, metrics, sample counts

Run:
    python3 src/train_multi_disease.py                  # defaults
    python3 src/train_multi_disease.py --epochs 5
    python3 src/train_multi_disease.py --max-per-class 2000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Ordered class list -> controls model output order and meta.json indexing.
CLASS_NAMES: List[str] = [
    "Normal",
    "Pneumonia",
    "COVID-19",
    "Tuberculosis",
    "Lung_Opacity",
]

# Each tuple: (label index, source directory relative to DATASETS_DIR)
CLASS_SOURCES: Dict[str, List[str]] = {
    "Normal": [
        "chest_xray/train/NORMAL",
        "COVID-19_Radiography_Dataset/Normal/images",
        "TB_Chest_Radiography_Database/Normal",
    ],
    "Pneumonia": [
        "chest_xray/train/PNEUMONIA",
        "COVID-19_Radiography_Dataset/Viral Pneumonia/images",
    ],
    "COVID-19": [
        "COVID-19_Radiography_Dataset/COVID/images",
    ],
    "Tuberculosis": [
        "TB_Chest_Radiography_Database/Tuberculosis",
    ],
    "Lung_Opacity": [
        "COVID-19_Radiography_Dataset/Lung_Opacity/images",
    ],
}

IMG_EXTS = {".png", ".jpg", ".jpeg"}
MODEL_OUT = MODELS_DIR / "multi_disease_model.pth"
META_OUT = MODELS_DIR / "multi_disease_meta.json"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def collect_samples(max_per_class: int, seed: int = 42) -> Tuple[List[str], List[int], Dict[str, int]]:
    """Walk the source folders and build a (file_path, label) list.

    Caps per-class count so the dataset doesn't get dominated by the large
    Normal category. Sampling is deterministic given the seed.
    """
    rng = random.Random(seed)
    paths: List[str] = []
    labels: List[int] = []
    per_class_counts: Dict[str, int] = {}

    for class_idx, class_name in enumerate(CLASS_NAMES):
        gathered: List[str] = []
        for rel_dir in CLASS_SOURCES[class_name]:
            abs_dir = DATASETS_DIR / rel_dir
            if not abs_dir.exists():
                print(f"  [warn] missing source dir for {class_name}: {abs_dir}")
                continue
            for root, _dirs, files in os.walk(abs_dir):
                # Skip "masks" subfolders in the COVID-19 dataset.
                if Path(root).name.lower() == "masks":
                    continue
                for fn in files:
                    if Path(fn).suffix.lower() in IMG_EXTS:
                        gathered.append(str(Path(root) / fn))

        rng.shuffle(gathered)
        if max_per_class and len(gathered) > max_per_class:
            gathered = gathered[:max_per_class]

        per_class_counts[class_name] = len(gathered)
        paths.extend(gathered)
        labels.extend([class_idx] * len(gathered))

    return paths, labels, per_class_counts


def validate_class_counts(per_class_counts: Dict[str, int], min_per_class: int = 2) -> None:
    """Fail early when any configured disease class cannot be trained/evaluated."""
    missing = [
        class_name
        for class_name in CLASS_NAMES
        if per_class_counts.get(class_name, 0) < min_per_class
    ]
    if missing:
        details = ", ".join(
            f"{class_name}={per_class_counts.get(class_name, 0)}"
            for class_name in missing
        )
        raise RuntimeError(
            "Not enough images for every multi-disease class. "
            f"Need at least {min_per_class} per class for stratified train/val split; "
            f"problem classes: {details}."
        )


class ChestXrayDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int], transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Fall back to a black image if a file is corrupt so one bad
            # sample doesn't crash a whole epoch.
            img = Image.new("RGB", (224, 224))
        return self.transform(img), label


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tf, eval_tf


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Freeze all, then unfreeze the last block + fc for faster/better
    # transfer learning than only retraining fc.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            if train and batch_idx % 20 == 0:
                print(
                    f"    batch {batch_idx}/{len(loader)}  loss={loss.item():.4f}"
                )

    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-per-class", type=int, default=1500,
                        help="Cap images per class (keeps training balanced).")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device()
    print(f"[device] {device}")

    print("[1/5] Collecting image paths from all three datasets")
    paths, labels, counts = collect_samples(max_per_class=args.max_per_class, seed=args.seed)
    total = len(paths)
    print(f"      total samples: {total:,}")
    for cls, n in counts.items():
        print(f"        {cls:<14} {n:,}")
    if total == 0:
        raise RuntimeError("No training images found. Check dataset folders.")
    validate_class_counts(counts)

    paths_train, paths_val, y_train, y_val = train_test_split(
        paths, labels, test_size=0.2, random_state=args.seed, stratify=labels
    )

    train_tf, eval_tf = build_transforms()
    train_ds = ChestXrayDataset(paths_train, y_train, train_tf)
    val_ds = ChestXrayDataset(paths_val, y_val, eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    print("\n[2/5] Building ResNet18 (pretrained, layer4+fc trainable)")
    model = build_model(num_classes=len(CLASS_NAMES)).to(device)

    class_counts = np.bincount(y_train, minlength=len(CLASS_NAMES)).astype(np.float32)
    # Inverse-frequency weights (then normalised).
    weights = class_counts.sum() / (len(CLASS_NAMES) * class_counts + 1e-6)
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"      class weights: {weights.round(3).tolist()}")

    criterion = nn.CrossEntropyLoss(weight=weights_t)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)

    print("\n[3/5] Training")
    best_val_acc = 0.0
    history = []
    for epoch in range(args.epochs):
        print(f"\n  Epoch {epoch + 1}/{args.epochs}")
        t0 = time.time()
        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc, val_preds, val_labels = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        dt = time.time() - t0
        print(
            f"    train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  ({dt:.1f}s)"
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "seconds": round(dt, 2),
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"    saved new best model -> {MODEL_OUT.name}")

    print("\n[4/5] Reloading best model for final evaluation")
    try:
        state = torch.load(MODEL_OUT, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(MODEL_OUT, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    _, _, final_preds, final_labels = run_epoch(
        model, val_loader, criterion, optimizer, device, train=False
    )

    acc = accuracy_score(final_labels, final_preds)
    macro_prec = precision_score(final_labels, final_preds, average="macro", zero_division=0)
    macro_rec = recall_score(final_labels, final_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(final_labels, final_preds, average="macro", zero_division=0)
    cm = confusion_matrix(final_labels, final_preds, labels=list(range(len(CLASS_NAMES)))).tolist()

    print("\n      Final validation metrics:")
    print(f"        accuracy        : {acc:.4f}")
    print(f"        precision(macro): {macro_prec:.4f}")
    print(f"        recall(macro)   : {macro_rec:.4f}")
    print(f"        f1(macro)       : {macro_f1:.4f}")
    print("\n      Classification report:")
    print(classification_report(final_labels, final_preds, target_names=CLASS_NAMES, digits=4))

    print("[5/5] Saving meta")
    meta = {
        "class_names": CLASS_NAMES,
        "class_sources": CLASS_SOURCES,
        "num_classes": len(CLASS_NAMES),
        "per_class_counts": counts,
        "n_train": len(paths_train),
        "n_val": len(paths_val),
        "max_per_class": args.max_per_class,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "history": history,
        "metrics": {
            "accuracy": round(float(acc), 4),
            "precision_macro": round(float(macro_prec), 4),
            "recall_macro": round(float(macro_rec), 4),
            "f1_macro": round(float(macro_f1), 4),
            "confusion_matrix": cm,
        },
    }
    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Model -> {MODEL_OUT}")
    print(f"      Meta  -> {META_OUT}")


if __name__ == "__main__":
    main()
