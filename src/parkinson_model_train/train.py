"""Frozen DINO encoder training for Parkinson drawing crops."""

import json
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from parkinson_model_train.data import (
    DatasetSplit,
    ImageRecord,
    TrainingConfig,
    labels_to_indices,
    load_manifest,
    split_by_subject,
    validate_image_paths,
    validate_split_labels,
)
from parkinson_model_train.evaluation import compute_metrics
from parkinson_model_train.reporting import write_metric_plots


def train_classifier(config: TrainingConfig) -> Path:
    """Train a classifier on frozen DINO embeddings and return the checkpoint path."""
    config.validate()
    _seed_training(config.seed)

    records = load_manifest(config.manifest_path)
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
    train_loader = _build_data_loader(
        split.train,
        label_to_index,
        train_transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        seed=config.seed,
    )
    val_loader = _build_data_loader(
        split.val,
        label_to_index,
        eval_transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        seed=None,
    )
    test_loader = _build_data_loader(
        split.test,
        label_to_index,
        eval_transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        seed=None,
    )

    import timm

    encoder = timm.create_model(config.model_name, pretrained=True, num_classes=0).to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    feature_count = encoder.num_features
    classifier = nn.Linear(feature_count, len(label_to_index)).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_val_metric = -1.0
    best_state_dict = None
    history = []
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "best_classifier.pt"

    for epoch in range(1, config.epochs + 1):
        train_loss = _train_epoch(encoder, classifier, train_loader, optimizer, loss_fn, device)
        val_metrics = _evaluate(encoder, classifier, val_loader, loss_fn, device, label_to_index) if split.val else {}
        val_acc_str = f"val_accuracy={val_metrics.get('accuracy', 0.0):.4f}"
        if "balanced_accuracy" in val_metrics:
            val_acc_str += f" val_balanced_accuracy={val_metrics['balanced_accuracy']:.4f}"
        print(f"epoch={epoch} train_loss={train_loss:.4f} {val_acc_str}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_metrics": val_metrics})

        metric_name = "balanced_accuracy" if "balanced_accuracy" in val_metrics else "accuracy"
        val_metric_val = val_metrics.get(metric_name, 0.0)

        if not split.val or val_metric_val >= best_val_metric:
            best_val_metric = val_metric_val
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
    test_metrics = _evaluate(encoder, classifier, test_loader, loss_fn, device, label_to_index) if split.test else {}

    summary = _write_run_summary(config.output_dir, label_to_index, split, test_metrics, history)
    write_metric_plots(config.output_dir, summary)
    return checkpoint_path


class DrawingDataset(Dataset):
    """PyTorch dataset for labelled drawing crops."""

    def __init__(
        self,
        records: list[ImageRecord],
        label_to_index: dict[str, int],
        transform: Any,
    ) -> None:
        self.records = records
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.label_to_index[record.label]


def build_image_transforms(config: TrainingConfig) -> tuple[Any, Any]:
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
    """Return augmentation settings written into checkpoints."""
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


def _build_data_loader(
    records: list[ImageRecord],
    label_to_index: dict[str, int],
    transform: Any,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int | None,
) -> DataLoader:
    generator = _torch_generator(seed) if seed is not None else None
    return DataLoader(
        DrawingDataset(records, label_to_index, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def _train_epoch(encoder, classifier, data_loader, optimizer, loss_fn, device) -> float:
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
    if total_items == 0:
        msg = "Training data loader produced no batches"
        raise ValueError(msg)
    return total_loss / total_items


def _evaluate(encoder, classifier, data_loader, loss_fn, device, label_to_index) -> dict[str, float]:
    classifier.eval()
    total_loss = 0.0
    total_items = 0
    y_true_list = []
    y_pred_list = []
    y_prob_list = []

    pos_index = 1
    if len(label_to_index) == 2:
        for lbl, idx in label_to_index.items():
            if lbl.upper() in {"PD", "PATIENT"}:
                pos_index = idx
                break

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="eval", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            features = encoder(images)
            logits = classifier(features)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            y_true_list.extend(labels.cpu().tolist())
            y_pred_list.extend(preds.cpu().tolist())
            if len(label_to_index) == 2:
                y_prob_list.extend(probs[:, pos_index].cpu().tolist())
            total_items += labels.size(0)

    if total_items == 0:
        msg = "Evaluation data loader produced no batches"
        raise ValueError(msg)

    metrics = {"loss": total_loss / total_items}
    if len(label_to_index) == 2:
        y_true_mapped = [1 if y == pos_index else 0 for y in y_true_list]
        y_pred_mapped = [1 if y == pos_index else 0 for y in y_pred_list]
        binary_metrics = compute_metrics(y_true_mapped, y_pred_mapped, y_prob_list)
        metrics.update(binary_metrics)
    else:
        correct = sum(1 for yt, yp in zip(y_true_list, y_pred_list, strict=True) if yt == yp)
        metrics["accuracy"] = correct / total_items if total_items > 0 else 0.0

    return metrics


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _seed_training(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _write_run_summary(
    output_dir: Path,
    label_to_index: dict[str, int],
    split: DatasetSplit,
    test_metrics: dict[str, float],
    history: list[dict[str, object]],
) -> dict[str, object]:
    summary = {
        "label_to_index": label_to_index,
        "split_counts": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        "subject_counts": {
            "train": len({record.subject_id for record in split.train}),
            "val": len({record.subject_id for record in split.val}),
            "test": len({record.subject_id for record in split.test}),
        },
        "test_metrics": test_metrics,
        "history": history,
    }
    Path(output_dir, "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
