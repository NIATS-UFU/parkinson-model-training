"""Dataset manifest loading and subject-level splitting utilities."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageRecord:
    """One labelled image and its subject identifier."""

    image_path: Path
    subject_id: str
    label: str


@dataclass(frozen=True)
class DatasetSplit:
    """Train, validation, and test records split by subject."""

    train: list[ImageRecord]
    val: list[ImageRecord]
    test: list[ImageRecord]


def load_manifest(
    manifest_path: Path,
    *,
    image_column: str = "image_path",
    subject_column: str = "subject_id",
    label_column: str = "label",
) -> list[ImageRecord]:
    """Load a CSV manifest with image path, subject, and label columns."""
    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing_columns = {image_column, subject_column, label_column} - set(reader.fieldnames or [])
        if missing_columns:
            joined = ", ".join(sorted(missing_columns))
            msg = f"Manifest is missing required columns: {joined}"
            raise ValueError(msg)

        records = []
        base_dir = manifest_path.parent
        for line_number, row in enumerate(reader, start=2):
            image_value = (row.get(image_column) or "").strip()
            subject_id = (row.get(subject_column) or "").strip()
            label = (row.get(label_column) or "").strip()
            if not image_value or not subject_id or not label:
                msg = f"Manifest row {line_number} has empty image, subject, or label"
                raise ValueError(msg)
            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = base_dir / image_path
            records.append(ImageRecord(image_path=image_path, subject_id=subject_id, label=label))

    if not records:
        msg = "Manifest contains no records"
        raise ValueError(msg)
    return records


def validate_image_paths(records: list[ImageRecord]) -> None:
    """Raise ValueError if any image path in the records is missing."""
    missing = [str(record.image_path) for record in records if not record.image_path.exists()]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", and {len(missing) - 5} more"
        msg = f"Missing image files: {preview}{suffix}"
        raise ValueError(msg)


def split_by_subject(
    records: list[ImageRecord],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> DatasetSplit:
    """Split records by subject to avoid image leakage between splits."""
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        msg = "val_fraction and test_fraction must be non-negative and sum to less than 1"
        raise ValueError(msg)

    subject_ids = sorted({record.subject_id for record in records})
    if len(subject_ids) < 3 and (val_fraction > 0 or test_fraction > 0):
        msg = "At least three subjects are required for train/validation/test splitting"
        raise ValueError(msg)

    shuffled_subjects = subject_ids[:]
    random.Random(seed).shuffle(shuffled_subjects)

    subject_count = len(shuffled_subjects)
    test_count = _fraction_to_count(subject_count, test_fraction)
    val_count = _fraction_to_count(subject_count, val_fraction)

    test_subjects = set(shuffled_subjects[:test_count])
    val_subjects = set(shuffled_subjects[test_count : test_count + val_count])
    train_subjects = set(shuffled_subjects[test_count + val_count :])

    train = [record for record in records if record.subject_id in train_subjects]
    val = [record for record in records if record.subject_id in val_subjects]
    test = [record for record in records if record.subject_id in test_subjects]

    if not train:
        msg = "Subject split produced an empty training set"
        raise ValueError(msg)
    return DatasetSplit(train=train, val=val, test=test)


def labels_to_indices(records: list[ImageRecord]) -> dict[str, int]:
    """Return deterministic class indices for manifest labels."""
    labels = sorted({record.label for record in records})
    if len(labels) < 2:
        msg = "At least two labels are required for classification"
        raise ValueError(msg)
    return {label: index for index, label in enumerate(labels)}


def validate_split_labels(split: DatasetSplit) -> None:
    """Raise ValueError if split labels cannot support classifier evaluation."""
    train_labels = {record.label for record in split.train}
    if len(train_labels) < 2:
        msg = "Training split must contain at least two labels"
        raise ValueError(msg)

    missing_labels = {
        split_name: sorted({record.label for record in records} - train_labels)
        for split_name, records in {"validation": split.val, "test": split.test}.items()
    }
    missing_labels = {split_name: labels for split_name, labels in missing_labels.items() if labels}
    if missing_labels:
        details = "; ".join(f"{split_name}: {', '.join(labels)}" for split_name, labels in missing_labels.items())
        msg = f"Split contains labels absent from training split: {details}"
        raise ValueError(msg)


def _fraction_to_count(total: int, fraction: float) -> int:
    if fraction == 0:
        return 0
    return max(1, round(total * fraction))
