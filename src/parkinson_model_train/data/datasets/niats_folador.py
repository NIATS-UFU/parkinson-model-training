"""Loader logic for the NIATS Folador drawing dataset."""

from pathlib import Path

from parkinson_model_train.data import ImageRecord
from parkinson_model_train.data.datasets.utils import build_records, drawing_type_from_name, iter_image_paths


def load_niats_folador_records(root: Path) -> list[ImageRecord]:
    """Load the Folador NIATS dataset from either combined or per-day folder layouts."""
    return build_records(
        root,
        iter_image_paths(root),
        dataset="niats-folador",
        subject_id_from_path=_folador_subject_id,
        metadata_from_path=_folador_metadata,
    )


def _folador_subject_id(image_path: Path, _root: Path) -> str:
    _, patient_id, drawing_code = _folador_name_parts(image_path)
    return f"{drawing_code[:2]}-{patient_id}"


def _folador_metadata(image_path: Path, root: Path, _label: str) -> dict[str, str]:
    day, _, drawing_code = _folador_name_parts(image_path)
    relative_parts = image_path.relative_to(root).parts
    return {
        "drawing_type": drawing_type_from_name("/".join(relative_parts)),
        "collection_day": day,
        "source_subset": _folador_subset(relative_parts),
        "split": _folador_split(relative_parts),
        "source_code": drawing_code,
    }


def _folador_name_parts(image_path: Path) -> tuple[str, str, str]:
    day, patient_id, drawing_code_with_crop = image_path.stem.split("_", maxsplit=2)
    drawing_code = drawing_code_with_crop[:2]
    return day, patient_id, drawing_code


def _folador_subset(relative_parts: tuple[str, ...]) -> str:
    if "dataset-junto" in relative_parts:
        return "dataset-junto"
    if "dataset-separado" in relative_parts:
        return "dataset-separado"
    return "unknown"


def _folador_split(relative_parts: tuple[str, ...]) -> str:
    for part in relative_parts:
        if part in {"treinamento", "teste"}:
            return part
    return "unknown"
