"""Shared utility functions for parsing dataset structures."""

import unicodedata
from collections.abc import Callable, Iterable
from pathlib import Path

from parkinson_model_train.data import ImageRecord, _is_image_file, _require_records


def normalize_text(text: str) -> str:
    """Normalize text by converting to lowercase and stripping accents/diacritics."""
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(character for character in normalized if not unicodedata.combining(character))


def label_from_text(text: str) -> str:
    """Infer PD (Parkinson's Disease) or HC (Healthy Control) labels from text segments."""
    normalized = normalize_text(text)
    if any(token in normalized for token in ["parkinson", "patient", "patients", "_p", "-p"]):
        return "PD"
    if any(token in normalized for token in ["healthy", "control", "controle", "higido", "higidos", "_h", "-h"]):
        return "HC"
    msg = f"Could not infer Parkinson/healthy label from: {text}"
    raise ValueError(msg)


def drawing_type_from_name(name: str) -> str:
    """Infer drawing type (spiral, meander, circle, sentence) from drawing filenames."""
    normalized = normalize_text(name)
    if any(token in normalized for token in ["spi", "spiral", "espiral", "sp"]):
        return "spiral"
    if any(token in normalized for token in ["mea", "meander", "onda", "ondas", "wave"]):
        return "meander"
    if any(token in normalized for token in ["circle", "circ"]):
        return "circle"
    if "sen" in normalized:
        return "sentence"
    return "unknown"


def iter_image_paths(root: Path, pattern: str = "**/*") -> Iterable[Path]:
    """Yield image files under ``root`` matching the given glob pattern."""
    for path in sorted(root.glob(pattern)):
        if _is_image_file(path):
            yield path


def build_records(
    root: Path,
    image_paths: Iterable[Path],
    *,
    dataset: str,
    subject_id_from_path: Callable[[Path, Path], str],
    metadata_from_path: Callable[[Path, Path, str], dict[str, str]] | None = None,
    label_from_path: Callable[[Path, Path], str] | None = None,
) -> list[ImageRecord]:
    """Build normalized records from image paths with dataset-specific hooks."""
    resolve_label = label_from_path or _label_from_relative_path
    records = []
    for image_path in image_paths:
        label = resolve_label(image_path, root)
        metadata = {
            "dataset": dataset,
            "source_label": label,
            "label_source": "filename",
        }
        if metadata_from_path is not None:
            metadata.update(metadata_from_path(image_path, root, label))
        records.append(
            ImageRecord(
                image_path=image_path,
                subject_id=subject_id_from_path(image_path, root),
                label=label,
                metadata=metadata,
            )
        )
    return _require_records(records, root)


def _label_from_relative_path(image_path: Path, root: Path) -> str:
    return label_from_text(str(image_path.relative_to(root)))


def load_image_tree_records(root: Path) -> list[ImageRecord]:
    """Load a generic image tree whose path names contain disease labels."""
    return build_records(
        root,
        iter_image_paths(root),
        dataset="image-tree",
        subject_id_from_path=_image_tree_subject_id,
        metadata_from_path=_image_tree_metadata,
    )


def _image_tree_subject_id(image_path: Path, _root: Path) -> str:
    return image_path.parent.name


def _image_tree_metadata(image_path: Path, _root: Path, _label: str) -> dict[str, str]:
    return {"drawing_type": drawing_type_from_name(image_path.name)}
