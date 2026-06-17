"""Frozen DINO encoder training for Parkinson drawing crops."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

from parkinson_model_train.config import TrainingConfig
from parkinson_model_train.data import (
    labels_to_indices,
    load_manifest,
    split_by_subject,
    validate_image_paths,
    validate_split_labels,
)

if TYPE_CHECKING:
    import torch


def train_classifier(config: TrainingConfig) -> Path:
    """Train a classifier on frozen DINO embeddings and return the checkpoint path."""
    config.validate()

    import timm
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    records = load_manifest(
        config.manifest_path,
        image_column=config.image_column,
        subject_column=config.subject_column,
        label_column=config.label_column,
    )
    validate_image_paths(records)
    label_to_index = labels_to_indices(records)
    split = split_by_subject(
        records,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )
    validate_split_labels(split)

    device = _resolve_device(config.device)
    train_transform, eval_transform = build_image_transforms(config)

    train_loader = DataLoader(
        DrawingDataset(split.train, label_to_index, train_transform),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        DrawingDataset(split.val, label_to_index, eval_transform),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        DrawingDataset(split.test, label_to_index, eval_transform),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    encoder = timm.create_model(config.model_name, pretrained=True, num_classes=0).to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    feature_count = encoder.num_features
    classifier = nn.Linear(feature_count, len(label_to_index)).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_val_accuracy = -1.0
    best_state_dict = None
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "best_classifier.pt"

    for epoch in range(1, config.epochs + 1):
        train_loss = _train_epoch(encoder, classifier, train_loader, optimizer, loss_fn, device)
        val_metrics = (
            _evaluate(encoder, classifier, val_loader, loss_fn, device) if split.val else {"loss": 0.0, "accuracy": 0.0}
        )
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_accuracy={val_metrics['accuracy']:.4f}")
        if not split.val or val_metrics["accuracy"] >= best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_state_dict = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}
            torch.save(
                {
                    "model_name": config.model_name,
                    "classifier_state_dict": classifier.state_dict(),
                    "label_to_index": label_to_index,
                    "image_size": config.image_size,
                    "augmentation": _augmentation_summary(config),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )

    if best_state_dict is not None:
        classifier.load_state_dict(best_state_dict)
    test_metrics = (
        _evaluate(encoder, classifier, test_loader, loss_fn, device) if split.test else {"loss": 0.0, "accuracy": 0.0}
    )
    _write_run_summary(config.output_dir, label_to_index, split, test_metrics)
    return checkpoint_path


class DrawingDataset:
    """PyTorch dataset for labelled drawing crops."""

    def __init__(self, records, label_to_index, transform):
        self.records = records
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        from PIL import Image

        record = self.records[index]
        with Image.open(record.image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.label_to_index[record.label]


def build_image_transforms(config: TrainingConfig) -> tuple[object, object]:
    """Build train and evaluation image transforms for DINO classifiers."""
    from torchvision import transforms

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    tensor_transforms = [transforms.ToTensor(), normalize]
    eval_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        *tensor_transforms,
    ])
    if not config.augment:
        return eval_transform, eval_transform

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            config.image_size,
            scale=(config.crop_scale_min, 1.0),
            ratio=(config.crop_ratio_min, config.crop_ratio_max),
        ),
        transforms.RandomRotation(config.rotation_degrees, fill=255),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=config.brightness_jitter, contrast=config.contrast_jitter)],
            p=config.jitter_probability,
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=config.blur_kernel_size, sigma=(0.1, 1.0))],
            p=config.blur_probability,
        ),
        *tensor_transforms,
    ])
    return train_transform, eval_transform


def _augmentation_summary(config: TrainingConfig) -> dict[str, bool | float | int]:
    return {
        "augment": config.augment,
        "rotation_degrees": config.rotation_degrees,
        "crop_scale_min": config.crop_scale_min,
        "crop_ratio_min": config.crop_ratio_min,
        "crop_ratio_max": config.crop_ratio_max,
        "blur_probability": config.blur_probability,
        "blur_kernel_size": config.blur_kernel_size,
        "jitter_probability": config.jitter_probability,
        "brightness_jitter": config.brightness_jitter,
        "contrast_jitter": config.contrast_jitter,
    }


def _train_epoch(encoder, classifier, data_loader, optimizer, loss_fn, device) -> float:
    import torch
    from tqdm.auto import tqdm

    classifier.train()
    total_loss = 0.0
    total_items = 0
    for images, labels in tqdm(data_loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            features = encoder(images)
        logits = classifier(features)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_items += labels.size(0)
    return total_loss / total_items


def _evaluate(encoder, classifier, data_loader, loss_fn, device) -> dict[str, float]:
    import torch
    from tqdm.auto import tqdm

    classifier.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="eval", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            features = encoder(images)
            logits = classifier(features)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_items += labels.size(0)
    return {"loss": total_loss / total_items, "accuracy": total_correct / total_items}


def _resolve_device(device: str) -> "torch.device":
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _write_run_summary(output_dir, label_to_index, split, test_metrics) -> None:
    summary = {
        "label_to_index": label_to_index,
        "split_counts": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        "subject_counts": {
            "train": len({record.subject_id for record in split.train}),
            "val": len({record.subject_id for record in split.val}),
            "test": len({record.subject_id for record in split.test}),
        },
        "test_metrics": test_metrics,
    }
    Path(output_dir, "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
