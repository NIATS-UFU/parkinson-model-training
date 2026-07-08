"""Manifest IO, image validation, and subject-safe splitting."""

import csv
import os
import random
from pathlib import Path

from parkinson_model_train.data.data_models import DatasetSplit, ImageRecord

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MANIFEST_SUFFIX = ".csv"
MANIFEST_FIELDS = ("image_path", "subject_id", "label")


def load_manifest(
    manifest_path: Path,
) -> list[ImageRecord]:
    """Load the project's canonical CSV manifest format."""
    if manifest_path.suffix.lower() != MANIFEST_SUFFIX:
        msg = f"Manifest must be a {MANIFEST_SUFFIX} file, got: {manifest_path.name}"
        raise ValueError(msg)

    records = _load_csv_manifest(manifest_path)
    if not records:
        msg = "Manifest contains no records"
        raise ValueError(msg)
    return records


def _load_csv_manifest(manifest_path: Path) -> list[ImageRecord]:
    """Parse normalized manifest rows into ``ImageRecord`` objects."""
    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = tuple(reader.fieldnames or ())
        missing_keys = sorted(set(MANIFEST_FIELDS) - set(fieldnames))
        if missing_keys:
            msg = f"CSV manifest is missing required columns: {', '.join(missing_keys)}"
            raise ValueError(msg)

        records = []
        for row_number, row in enumerate(reader, start=2):
            image_value = row.get("image_path")
            subject_value = row.get("subject_id")
            label_value = row.get("label")
            if not image_value or not subject_value or not label_value:
                msg = f"CSV manifest row {row_number} has an empty required value"
                raise ValueError(msg)

            records.append(_record_from_manifest_row(row, manifest_path))
    return records


def write_manifest(records: list[ImageRecord], manifest_path: Path) -> None:
    """Write records to the project's canonical CSV manifest format."""
    if not records:
        msg = "Cannot write an empty manifest"
        raise ValueError(msg)

    if manifest_path.suffix.lower() != MANIFEST_SUFFIX:
        msg = f"Manifest must be a {MANIFEST_SUFFIX} file, got: {manifest_path.name}"
        raise ValueError(msg)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv_manifest(records, manifest_path)


def _write_csv_manifest(records: list[ImageRecord], manifest_path: Path) -> None:
    """Write records with canonical columns first and metadata columns after."""
    metadata_keys = sorted({key for record in records for key in record.metadata})
    fieldnames = [*MANIFEST_FIELDS, *metadata_keys]
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(_manifest_row(record, manifest_path.parent))


def load_dataset_records(dataset: str, root: Path) -> list[ImageRecord]:
    """Load supported local dataset layouts into drawing image records."""
    from parkinson_model_train.data.datasets import DATASET_LOADERS

    try:
        loader = DATASET_LOADERS[dataset]
    except KeyError as error:
        msg = f"Unsupported dataset: {dataset}"
        raise ValueError(msg) from error
    return loader(root)


def validate_image_paths(records: list[ImageRecord]) -> None:
    """Raise ValueError if any image path in the records is missing."""
    missing = [str(record.image_path) for record in records if not record.image_path.exists()]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", and {len(missing) - 5} more"
        msg = f"Missing image files: {preview}{suffix}"
        raise ValueError(msg)


def validate_subject_labels(records: list[ImageRecord]) -> None:
    """Raise ValueError if any subject has images with conflicting labels."""
    conflicts: dict[str, set[str]] = {}
    for record in records:
        conflicts.setdefault(record.subject_id, set()).add(record.label)
    bad = {sid: labels for sid, labels in conflicts.items() if len(labels) > 1}
    if bad:
        details = "; ".join(f"{sid}: {sorted(labels)}" for sid, labels in sorted(bad.items()))
        msg = f"Subjects with conflicting labels: {details}"
        raise ValueError(msg)


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith("._")


def _manifest_image_path(image_path: Path, manifest_dir: Path) -> str:
    # Store relative paths so manifests remain portable within the workspace.
    absolute_image_path = image_path if image_path.is_absolute() else image_path.resolve()
    absolute_manifest_dir = manifest_dir if manifest_dir.is_absolute() else manifest_dir.resolve()
    return os.path.relpath(absolute_image_path, absolute_manifest_dir)


def _record_from_manifest_row(row: dict[str, str | None], manifest_path: Path) -> ImageRecord:
    """Convert one normalized CSV row into an ``ImageRecord``."""
    image_path = Path(row["image_path"] or "")
    if not image_path.is_absolute():
        image_path = (manifest_path.parent / image_path).resolve()

    metadata = {
        str(key): str(value) for key, value in row.items() if key not in MANIFEST_FIELDS and value not in {None, ""}
    }
    return ImageRecord(
        image_path=image_path,
        subject_id=str(row["subject_id"]),
        label=str(row["label"]),
        metadata=metadata,
    )


def _manifest_row(record: ImageRecord, manifest_dir: Path) -> dict[str, str]:
    """Render one ``ImageRecord`` into the normalized CSV row shape."""
    row = {
        "image_path": _manifest_image_path(record.image_path, manifest_dir),
        "subject_id": record.subject_id,
        "label": record.label,
    }
    row.update(record.metadata)
    return row


def _require_records(records: list[ImageRecord], root: Path) -> list[ImageRecord]:
    if not records:
        msg = f"No image records found under: {root}"
        raise ValueError(msg)
    return records


def split_by_subject(
    records: list[ImageRecord],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> DatasetSplit:
    """Split records by subject to avoid image leakage between splits, stratified by class label."""
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        msg = "val_fraction and test_fraction must be non-negative and sum to less than 1"
        raise ValueError(msg)

    subject_ids = sorted({record.subject_id for record in records})
    if len(subject_ids) < 3 and (val_fraction > 0 or test_fraction > 0):
        msg = "At least three subjects are required for train/validation/test splitting"
        raise ValueError(msg)

    subject_to_label = {}
    for record in records:
        existing_label = subject_to_label.setdefault(record.subject_id, record.label)
        if existing_label != record.label:
            msg = f"Subject {record.subject_id} has conflicting labels: {existing_label}, {record.label}"
            raise ValueError(msg)

    subjects_by_label = {}
    for subj_id, label in subject_to_label.items():
        subjects_by_label.setdefault(label, []).append(subj_id)

    for label in subjects_by_label:
        subjects_by_label[label].sort()

    train_subjects = set()
    val_subjects = set()
    test_subjects = set()

    rng = random.Random(seed)
    for _label, shuffled_subjects in sorted(subjects_by_label.items()):
        rng.shuffle(shuffled_subjects)
        n = len(shuffled_subjects)
        test_count = _fraction_to_count(n, test_fraction)
        val_count = _fraction_to_count(n, val_fraction)

        test_subjects.update(shuffled_subjects[:test_count])
        val_subjects.update(shuffled_subjects[test_count : test_count + val_count])
        train_subjects.update(shuffled_subjects[test_count + val_count :])

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

    incomplete_splits = {
        split_name: sorted(train_labels - {record.label for record in records})
        for split_name, records in {"validation": split.val, "test": split.test}.items()
        if records and {record.label for record in records} != train_labels
    }
    if incomplete_splits:
        details = "; ".join(f"{split_name}: {', '.join(labels)}" for split_name, labels in incomplete_splits.items())
        msg = f"Split is missing training labels: {details}"
        raise ValueError(msg)


def _fraction_to_count(total: int, fraction: float) -> int:
    if fraction == 0:
        return 0
    return max(1, round(total * fraction))
