"""Loader logic for the clinical Hemato drawing dataset."""

from pathlib import Path

from parkinson_model_train.data import ImageRecord
from parkinson_model_train.data.datasets.utils import build_records, drawing_type_from_name, iter_image_paths


def load_hemato_records(root: Path) -> list[ImageRecord]:
    """Load Hemato final crop folders produced by spiral-cropper."""
    return build_records(
        root,
        iter_image_paths(root, "**/5_final/*"),
        dataset="hemato",
        subject_id_from_path=_hemato_subject_id,
        metadata_from_path=_hemato_metadata,
    )


def _hemato_subject_id(image_path: Path, root: Path) -> str:
    source_dir = image_path.parent.parent
    relative_parts = source_dir.relative_to(root).parts
    return relative_parts[0] if len(relative_parts) > 1 else source_dir.name


def _hemato_metadata(image_path: Path, _root: Path, _label: str) -> dict[str, str]:
    source_dir = image_path.parent.parent
    return {
        "drawing_type": drawing_type_from_name(image_path.name),
        "drawing_session": source_dir.name,
    }
