"""Dataset manifest loading and subject-level splitting utilities."""

from __future__ import annotations

import csv
import os
import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    """One labelled image and its subject identifier."""

    image_path: Path
    subject_id: str
    label: str
    metadata: dict[str, str] = field(default_factory=dict)


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
                image_path = (base_dir / image_path).resolve()
            metadata = {
                key: value.strip()
                for key, value in row.items()
                if key not in {image_column, subject_column, label_column} and value is not None and value.strip()
            }
            records.append(ImageRecord(image_path=image_path, subject_id=subject_id, label=label, metadata=metadata))

    if not records:
        msg = "Manifest contains no records"
        raise ValueError(msg)
    return records


def write_manifest(records: list[ImageRecord], manifest_path: Path) -> None:
    """Write image records to a CSV manifest consumable by the training CLI."""
    if not records:
        msg = "Cannot write an empty manifest"
        raise ValueError(msg)

    metadata_columns = sorted({key for record in records for key in record.metadata})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "subject_id", "label", *metadata_columns])
        writer.writeheader()
        for record in records:
            row = {
                "image_path": _manifest_image_path(record.image_path, manifest_path.parent),
                "subject_id": record.subject_id,
                "label": record.label,
            }
            row.update(record.metadata)
            writer.writerow(row)


def load_dataset_records(dataset: str, root: Path) -> list[ImageRecord]:
    """Load supported local dataset layouts into training image records."""
    loaders = {
        "hemato": load_hemato_records,
        "handpd": load_handpd_records,
        "newhandpd": load_newhandpd_records,
        "kaggle-drawings": load_kaggle_drawings_records,
        "kaggle-spiral": load_kaggle_spiral_records,
        "roboflow": load_roboflow_records,
        "folador": load_folador_records,
        "image-tree": load_image_tree_records,
    }
    try:
        loader = loaders[dataset]
    except KeyError as error:
        msg = f"Unsupported dataset: {dataset}"
        raise ValueError(msg) from error
    return loader(root)


def load_hemato_records(root: Path) -> list[ImageRecord]:
    """Load Hemato final crop folders produced by spiral-cropper."""
    records = []
    for image_path in sorted(root.glob("**/5_final/*")):
        if not _is_image_file(image_path):
            continue
        source_dir = image_path.parent.parent
        label = _label_from_text(str(source_dir))
        records.append(
            ImageRecord(
                image_path=image_path,
                subject_id=source_dir.name,
                label=label,
                metadata={
                    "dataset": "hemato",
                    "drawing_type": _drawing_type_from_name(image_path.name),
                    "source_label": label,
                    "label_source": "filename",
                },
            )
        )
    return _require_records(records, root)


def load_handpd_records(root: Path) -> list[ImageRecord]:
    """Load original HandPD spiral and meander image folders."""
    records = []
    for image_path in sorted(root.glob("*/*")):
        if not _is_image_file(image_path) or image_path.parent.name == "__MACOSX":
            continue
        label = _label_from_text(image_path.parent.name)
        records.append(
            ImageRecord(
                image_path=image_path,
                subject_id=image_path.stem.split("-")[0],
                label=label,
                metadata={
                    "dataset": "handpd",
                    "drawing_type": _drawing_type_from_name(image_path.parent.name),
                    "source_label": label,
                    "label_source": "filename",
                },
            )
        )
    return _require_records(records, root)


def load_newhandpd_records(root: Path) -> list[ImageRecord]:
    """Load NewHandPD image folders for circles, meanders, and spirals."""
    records = []
    for image_path in sorted(root.glob("*/*")):
        if not _is_image_file(image_path) or image_path.parent.name == "__MACOSX":
            continue
        label = _label_from_text(image_path.name)
        records.append(
            ImageRecord(
                image_path=image_path,
                subject_id=_newhandpd_subject_id(image_path.name),
                label=label,
                metadata={
                    "dataset": "newhandpd",
                    "drawing_type": _drawing_type_from_name(image_path.name),
                    "source_label": label,
                    "label_source": "filename",
                },
            )
        )
    return _require_records(records, root)


def load_kaggle_spiral_records(root: Path) -> list[ImageRecord]:
    """Load Kaggle spiral-test folders labelled by directory name."""
    records = []
    for image_path in sorted(root.glob("*/*")):
        if not _is_image_file(image_path):
            continue
        label = _label_from_text(image_path.parent.name)
        records.append(
            ImageRecord(
                image_path=image_path,
                subject_id=f"{label.lower()}-{image_path.stem}",
                label=label,
                metadata={
                    "dataset": "kaggle-spiral",
                    "drawing_type": "spiral",
                    "source_label": label,
                    "label_source": "filename",
                    "source_dir": image_path.parent.name,
                },
            )
        )
    return _require_records(records, root)


def load_kaggle_drawings_records(root: Path) -> list[ImageRecord]:
    """Load Kaggle kmader/parkinsons-drawings spiral/wave image folders."""
    records = []
    for image_path in sorted(root.glob("**/*")):
        if not _is_image_file(image_path):
            continue
        relative_path = image_path.relative_to(root)
        relative_text = str(relative_path)
        label = _label_from_text(relative_text)
        drawing_type = _drawing_type_from_name(relative_text)
        records.append(
            ImageRecord(
                image_path=image_path,
                subject_id=_kaggle_drawings_subject_id(relative_path, label, drawing_type),
                label=label,
                metadata={
                    "dataset": "kaggle-drawings",
                    "kaggle_slug": "kmader/parkinsons-drawings",
                    "drawing_type": drawing_type,
                    "source_label": label,
                    "label_source": "path",
                    "source_split": _source_split_from_parts(relative_path.parts),
                },
            )
        )
    return _require_records(records, root)


def load_roboflow_records(root: Path) -> list[ImageRecord]:
    """Load Roboflow classification exports split by train/valid/test and class folders."""
    records = []
    for image_path in sorted(root.glob("*/*/*/*")):
        if not _is_image_file(image_path):
            continue
        relative_parts = image_path.relative_to(root).parts
        if len(relative_parts) != 4:
            continue
        source_split, drawing_dir, label_dir, _ = relative_parts
        label = _label_from_text(label_dir)
        original_stem = _roboflow_original_stem(image_path)
        records.append(
            ImageRecord(
                image_path=image_path,
                subject_id=original_stem,
                label=label,
                metadata={
                    "dataset": "roboflow",
                    "drawing_type": _drawing_type_from_name(drawing_dir),
                    "source_label": label,
                    "label_source": "path",
                    "source_split": source_split,
                    "original_stem": original_stem,
                },
            )
        )
    return _require_records(records, root)


def load_folador_records(root: Path) -> list[ImageRecord]:
    """Load Folador HOG tremor-detection image folders from dataset-junto."""
    dataset_roots = [path for path in [root, *root.rglob("dataset-junto")] if path.name == "dataset-junto"]
    search_roots = dataset_roots or [root]
    records = []
    for dataset_root in search_roots:
        for image_path in sorted(dataset_root.glob("*/*/*/*")):
            if not _is_image_file(image_path):
                continue
            relative_parts = image_path.relative_to(dataset_root).parts
            if len(relative_parts) != 4:
                continue
            drawing_dir, source_split, label_dir, _ = relative_parts
            label = _label_from_text(label_dir)
            records.append(
                ImageRecord(
                    image_path=image_path,
                    subject_id=_folador_subject_id(image_path.name),
                    label=label,
                    metadata={
                        "dataset": "folador",
                        "drawing_type": _drawing_type_from_name(drawing_dir),
                        "source_label": label,
                        "label_source": "path",
                        "source_split": source_split,
                    },
                )
            )
    return _require_records(records, root)


def load_image_tree_records(root: Path) -> list[ImageRecord]:
    """Load a generic image tree whose path names contain disease labels."""
    records = []
    for image_path in sorted(root.glob("**/*")):
        if not _is_image_file(image_path):
            continue
        label = _label_from_text(str(image_path.relative_to(root)))
        records.append(
            ImageRecord(
                image_path=image_path,
                subject_id=image_path.parent.name,
                label=label,
                metadata={
                    "dataset": "image-tree",
                    "drawing_type": _drawing_type_from_name(image_path.name),
                    "source_label": label,
                    "label_source": "filename",
                },
            )
        )
    return _require_records(records, root)


def validate_image_paths(records: list[ImageRecord]) -> None:
    """Raise ValueError if any image path in the records is missing."""
    missing = [str(record.image_path) for record in records if not record.image_path.exists()]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", and {len(missing) - 5} more"
        msg = f"Missing image files: {preview}{suffix}"
        raise ValueError(msg)


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith("._")


def _manifest_image_path(image_path: Path, manifest_dir: Path) -> str:
    absolute_image_path = image_path if image_path.is_absolute() else image_path.resolve()
    absolute_manifest_dir = manifest_dir if manifest_dir.is_absolute() else manifest_dir.resolve()
    return os.path.relpath(absolute_image_path, absolute_manifest_dir)


def _require_records(records: list[ImageRecord], root: Path) -> list[ImageRecord]:
    if not records:
        msg = f"No image records found under: {root}"
        raise ValueError(msg)
    return records


def _label_from_text(text: str) -> str:
    normalized = _normalize_text(text)
    if any(token in normalized for token in ["parkinson", "patient", "patients", "_p", "-p"]):
        return "PD"
    if any(token in normalized for token in ["healthy", "control", "controle", "higido", "higidos", "_h", "-h"]):
        return "HC"
    msg = f"Could not infer Parkinson/healthy label from: {text}"
    raise ValueError(msg)


def _drawing_type_from_name(name: str) -> str:
    normalized = _normalize_text(name)
    if any(token in normalized for token in ["spi", "spiral", "espiral", "sp"]):
        return "spiral"
    if any(token in normalized for token in ["mea", "meander", "onda", "ondas", "wave"]):
        return "meander"
    if any(token in normalized for token in ["circle", "circ"]):
        return "circle"
    if "sen" in normalized:
        return "sentence"
    return "unknown"


def _newhandpd_subject_id(file_name: str) -> str:
    match = re.search(r"-([HP]\d+)\.", file_name, flags=re.IGNORECASE)
    if not match:
        msg = f"Could not infer NewHandPD subject from: {file_name}"
        raise ValueError(msg)
    return match.group(1).upper()


def _roboflow_original_stem(path: Path) -> str:
    stem = path.stem.split(".rf.", maxsplit=1)[0]
    return re.sub(r"_png$", "", stem, flags=re.IGNORECASE)


def _kaggle_drawings_subject_id(relative_path: Path, label: str, drawing_type: str) -> str:
    source_stem = str(relative_path.with_suffix(""))
    safe_stem = re.sub(r"[^a-zA-Z0-9]+", "-", source_stem).strip("-").lower()
    return f"{label.lower()}-{drawing_type}-{safe_stem}"


def _source_split_from_parts(parts: tuple[str, ...]) -> str:
    for part in parts:
        normalized = _normalize_text(part)
        if normalized in {"train", "training"}:
            return "train"
        if normalized in {"valid", "validation", "val"}:
            return "valid"
        if normalized in {"test", "testing"}:
            return "test"
    return "unknown"


def _folador_subject_id(file_name: str) -> str:
    match = re.search(r"([HP][EO]\d+)", file_name, flags=re.IGNORECASE)
    if not match:
        msg = f"Could not infer Folador subject from: {file_name}"
        raise ValueError(msg)
    return match.group(1).upper()


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(character for character in normalized if not unicodedata.combining(character))


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
