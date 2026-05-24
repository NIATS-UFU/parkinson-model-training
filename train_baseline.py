"""Patient-Stratified Cross-Validation Training Baseline for Parkinson Drawings.

This script implements a leakage-safe patient-level stratified Group K-Fold split
on spiral number 3 (test3_spi_centered.png) from the Hemato internal dataset.
It trains a model using standard PyTorch torchvision augmentations and evaluates
consolidated cross-validation performance.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from parkinson_model_train.data import load_hemato_records


# -------------------------------------------------------------------------
# Simple CNN Architecture for fast CPU baseline testing
# -------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """Lightweight CNN that trains very quickly from scratch on CPU."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 64-dim output
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# -------------------------------------------------------------------------
# PyTorch Dataset
# -------------------------------------------------------------------------
class SpiralBaselineDataset(Dataset):
    """PyTorch dataset for loading and transforming spiral drawings."""

    def __init__(self, records: list, label_to_index: dict[str, int], transform: transforms.Compose) -> None:
        self.records = records
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(record.image_path) as img:
            rgb_image = img.convert("RGB")
            tensor = self.transform(rgb_image)
        return tensor, self.label_to_index[record.label]


# -------------------------------------------------------------------------
# Stratified Group K-Fold splitting logic (Pure Python/Deterministic)
# -------------------------------------------------------------------------
def stratified_group_k_fold(records: list, num_folds: int = 5, seed: int = 42) -> list[dict[str, list]]:
    """Split records into training and validation folds safely at the patient-level.

    Guarantees that drawings from a patient reside in either the train split or
    validation split, never both, while keeping the ratio of PD vs HC balanced.
    """
    # 1. Map drawings to their patient IDs
    patient_to_records = {}
    patient_to_label = {}
    for record in records:
        # patient_id is the grandparent directory name (e.g. V1)
        patient_id = record.image_path.parent.parent.parent.name
        patient_to_records.setdefault(patient_id, []).append(record)
        patient_to_label[patient_id] = record.label

    # 2. Segregate patient IDs by label
    pd_patients = sorted([p for p, label in patient_to_label.items() if label == "PD"])
    hc_patients = sorted([p for p, label in patient_to_label.items() if label == "HC"])

    # 3. Shuffle patients deterministically using random.Random
    rng = random.Random(seed)
    rng.shuffle(pd_patients)
    rng.shuffle(hc_patients)

    # 4. Distribute patients across folds in a round-robin fashion
    folds = [{"train_patients": [], "val_patients": []} for _ in range(num_folds)]
    for idx, p in enumerate(pd_patients):
        folds[idx % num_folds]["val_patients"].append(p)
    for idx, p in enumerate(hc_patients):
        folds[idx % num_folds]["val_patients"].append(p)

    # 5. Populate train patients as the complement of validation patients
    all_patients = set(patient_to_records.keys())
    for f in folds:
        val_set = set(f["val_patients"])
        f["train_patients"] = sorted(list(all_patients - val_set))

    # 6. Map patient IDs back to their corresponding image records
    splits = []
    for f in folds:
        train_records = []
        for p in f["train_patients"]:
            train_records.extend(patient_to_records[p])

        val_records = []
        for p in f["val_patients"]:
            val_records.extend(patient_to_records[p])

        splits.append({"train": train_records, "val": val_records})

    return splits


# -------------------------------------------------------------------------
# Metrics Calculation (Pure Python/No scikit-learn dependency)
# -------------------------------------------------------------------------
def compute_roc_auc(y_true: list[int], y_scores: list[float]) -> float:
    """Compute the Area Under the Receiver Operating Characteristic Curve (ROC AUC)."""
    paired = list(zip(y_true, y_scores, strict=True))
    paired.sort(key=lambda x: x[1], reverse=True)

    n_pos = sum(1 for y, _ in paired if y == 1)
    n_neg = len(paired) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    auc = 0.0
    fp, tp = 0, 0
    prev_fp, prev_tp = 0, 0
    prev_score = float("-inf")

    for y, score in paired:
        if score != prev_score:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
            prev_fp = fp
            prev_tp = tp
            prev_score = score
        if y == 1:
            tp += 1
        else:
            fp += 1

    auc += (n_neg - prev_fp) * (n_pos + prev_tp) / 2.0
    return auc / (n_pos * n_neg)


def compute_fold_metrics(y_true: list[int], y_pred: list[int], y_prob: list[float]) -> dict[str, float]:
    """Compute Sensitivity, Specificity, Balanced Accuracy, and ROC AUC."""
    tp, fp, tn, fn = 0, 0, 0, 0
    for yt, yp in zip(y_true, y_pred, strict=True):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 1 and yp == 0:
            fn += 1

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2.0
    roc_auc = compute_roc_auc(y_true, y_prob)

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_acc,
        "roc_auc": roc_auc,
        "accuracy": (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0,
    }


# -------------------------------------------------------------------------
# Training Function
# -------------------------------------------------------------------------
def run_training() -> None:
    """Parse arguments and train the patient-stratified cross-validation baseline."""
    parser = argparse.ArgumentParser(description="Train a leakage-safe patient-stratified baseline on spiral number 3.")
    parser.add_argument("--data-dir", type=str, default="data/hemato", help="Path to hemato dataset directory.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs per fold.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for DataLoader.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--model-type",
        type=str,
        default="simple-cnn",
        choices=["simple-cnn", "resnet18", "vit_small_patch14_dinov2"],
        help="Model architecture backbone.",
    )
    parser.add_argument("--frozen-backbone", action="store_true", help="Freeze backbones and only train classifier.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility.")
    parser.add_argument("--num-folds", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--image-size", type=int, default=224, help="Size to resize drawings for training.")
    args = parser.parse_args()

    # Resolve device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. Discover data
    print(f"Scanning dataset in: {args.data_dir}...")
    records = load_hemato_records(Path(args.data_dir))
    # Filter for spiral number 3 only (test3_spi_centered.png)
    spiral3_records = [r for r in records if r.image_path.name == "test3_spi_centered.png"]
    print(f"Total drawings found (overall hemato): {len(records)}")
    print(f"Total spiral 3 drawings selected: {len(spiral3_records)}")

    # Get label index mapping
    label_to_index = {"HC": 0, "PD": 1}

    # 2. Get splits
    splits = stratified_group_k_fold(spiral3_records, num_folds=args.num_folds, seed=args.seed)

    # Print baseline stats
    patient_to_label = {}
    for r in spiral3_records:
        pid = r.image_path.parent.parent.parent.name
        patient_to_label[pid] = r.label
    p_labels = list(patient_to_label.values())
    print(f"Patient distribution: {p_labels.count('PD')} PD, {p_labels.count('HC')} HC")

    # Define transforms (PyTorch canonical example augmentations)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomRotation(15, fill=255),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    fold_results = []
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)

    for fold_idx in range(args.num_folds):
        print(f"\n--- FOLD {fold_idx + 1}/{args.num_folds} ---")
        split = splits[fold_idx]
        train_records = split["train"]
        val_records = split["val"]

        # Log patient composition
        train_patients = {r.image_path.parent.parent.parent.name for r in train_records}
        val_patients = {r.image_path.parent.parent.parent.name for r in val_records}
        print(f"Train set: {len(train_records)} drawings from {len(train_patients)} patients")
        print(f"Val set: {len(val_records)} drawings from {len(val_patients)} patients")
        assert train_patients.isdisjoint(val_patients), "Patient leakage detected!"

        # Create DataLoaders
        train_dataset = SpiralBaselineDataset(train_records, label_to_index, train_transform)
        val_dataset = SpiralBaselineDataset(val_records, label_to_index, val_transform)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Initialize model
        if args.model_type == "simple-cnn":
            model = SimpleCNN(num_classes=2)
        else:
            model_name = "resnet18" if args.model_type == "resnet18" else "vit_small_patch14_dinov2"
            print(f"Loading pretrained backbone: {model_name}...")
            if args.frozen_backbone:
                model = timm.create_model(model_name, pretrained=True, num_classes=0)
                # Freeze backbone parameters
                for param in model.parameters():
                    param.requires_grad = False
                # Wrap with a linear classifier head
                feature_dim = model.num_features
                model = nn.Sequential(model, nn.Linear(feature_dim, 2))
            else:
                model = timm.create_model(model_name, pretrained=True, num_classes=2)

        model = model.to(device)

        # Optimizer and Loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        best_val_balanced_acc = -1.0
        best_metrics = {}

        # Epoch loop
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            total_items = 0

            # Training epoch
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False):
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = loss_fn(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * labels.size(0)
                total_items += labels.size(0)

            avg_train_loss = train_loss / total_items

            # Evaluation
            model.eval()
            val_loss = 0.0
            y_true_list = []
            y_pred_list = []
            y_prob_list = []

            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch} Val", leave=False):
                    images = images.to(device)
                    labels = labels.to(device)

                    logits = model(images)
                    loss = loss_fn(logits, labels)
                    val_loss += loss.item() * labels.size(0)

                    probs = torch.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1)

                    y_true_list.extend(labels.cpu().tolist())
                    y_pred_list.extend(preds.cpu().tolist())
                    # Positive class prob (PD is index 1)
                    y_prob_list.extend(probs[:, 1].cpu().tolist())

            avg_val_loss = val_loss / len(val_dataset)
            metrics = compute_fold_metrics(y_true_list, y_pred_list, y_prob_list)

            print(
                f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {metrics['accuracy']:.4f} | Val Balanced Acc: {metrics['balanced_accuracy']:.4f}"
            )

            # Save best checkpoint for this fold
            if metrics["balanced_accuracy"] > best_val_balanced_acc:
                best_val_balanced_acc = metrics["balanced_accuracy"]
                best_metrics = metrics
                torch.save(
                    {
                        "fold": fold_idx + 1,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "metrics": metrics,
                    },
                    runs_dir / f"baseline_fold_{fold_idx + 1}.pt",
                )

        print(f"Best Fold {fold_idx + 1} Metrics:")
        print(f"  Accuracy:          {best_metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {best_metrics['balanced_accuracy']:.4f}")
        print(f"  Sensitivity (PD):  {best_metrics['sensitivity']:.4f}")
        print(f"  Specificity (HC):  {best_metrics['specificity']:.4f}")
        print(f"  ROC-AUC:           {best_metrics['roc_auc']:.4f}")
        fold_results.append(best_metrics)

    # -------------------------------------------------------------------------
    # Aggregate and consolidated CV metrics report
    # -------------------------------------------------------------------------
    print("\n=========================================================================")
    print("CONSOLIDATED CROSS-VALIDATION BASELINE PERFORMANCE")
    print("=========================================================================")
    avg_accuracy = sum(m["accuracy"] for m in fold_results) / args.num_folds
    avg_balanced_accuracy = sum(m["balanced_accuracy"] for m in fold_results) / args.num_folds
    avg_sensitivity = sum(m["sensitivity"] for m in fold_results) / args.num_folds
    avg_specificity = sum(m["specificity"] for m in fold_results) / args.num_folds
    avg_roc_auc = sum(m["roc_auc"] for m in fold_results) / args.num_folds

    print(f"Average Accuracy:          {avg_accuracy:.4f}")
    print(f"Average Balanced Accuracy: {avg_balanced_accuracy:.4f}")
    print(f"Average Sensitivity (PD):  {avg_sensitivity:.4f}")
    print(f"Average Specificity (HC):  {avg_specificity:.4f}")
    print(f"Average ROC-AUC:           {avg_roc_auc:.4f}")
    print("=========================================================================")


if __name__ == "__main__":
    run_training()
