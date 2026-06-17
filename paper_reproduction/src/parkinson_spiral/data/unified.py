"""Unified manifest schema + assembler.

One row per original image. No augmented rows here — augmentation is on-the-fly
in `dataset.PreprocessedSpiralDataset`.

Dataset names follow the paper: kaggle, folador (Folador et al. 2021), hemato.
"""

import hashlib
from pathlib import Path

import pandas as pd

MANIFEST_COLUMNS = [
    "image_id",
    "dataset",
    "subject_id",
    "drawing_type",
    "class_label",
    "published_split",
    "cv_fold",
    "session_id",
    "file_path",
    "source_notes",
]

MANIFEST_DTYPES = {
    "image_id": "string",
    "dataset": "string",
    "subject_id": "string",
    "drawing_type": "string",
    "class_label": "string",
    "published_split": "string",
    "cv_fold": "Int64",
    "session_id": "string",
    "file_path": "string",
    "source_notes": "string",
}

VALID_DATASETS = {"kaggle", "folador", "hemato"}
VALID_DRAWING_TYPES = {"spiral", "wave"}
VALID_LABELS = {"healthy", "parkinson"}


def make_image_id(dataset: str, rel_path: str) -> str:
    """Stable short hash so manifest rows survive path-root changes."""
    h = hashlib.sha1(f"{dataset}:{rel_path}".encode()).hexdigest()
    return f"{dataset}:{h[:16]}"


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate columns, set dtypes, enforce enum constraints."""
    missing = set(MANIFEST_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    df = df[MANIFEST_COLUMNS].copy()
    for col, dtype in MANIFEST_DTYPES.items():
        df[col] = df[col].astype(dtype)

    bad_ds = set(df["dataset"].dropna().unique()) - VALID_DATASETS
    if bad_ds:
        raise ValueError(f"invalid dataset values: {bad_ds}")
    bad_dt = set(df["drawing_type"].dropna().unique()) - VALID_DRAWING_TYPES
    if bad_dt:
        raise ValueError(f"invalid drawing_type values: {bad_dt}")
    bad_lbl = set(df["class_label"].dropna().unique()) - VALID_LABELS
    if bad_lbl:
        raise ValueError(f"invalid class_label values: {bad_lbl}")
    if df["subject_id"].isna().any():
        raise ValueError("subject_id has NaN — subject-level splitting requires this populated everywhere")
    if df["image_id"].duplicated().any():
        dupes = df.loc[df["image_id"].duplicated(), "image_id"].head(5).tolist()
        raise ValueError(f"duplicate image_id values (e.g. {dupes})")
    return df


def build_unified_manifest(data_root: Path) -> pd.DataFrame:
    """Concatenate per-dataset manifests; downstream callers add cv_fold."""
    from .folador import build_folador_manifest
    from .hemato import build_hemato_manifest
    from .kaggle import build_kaggle_manifest

    parts = [
        build_kaggle_manifest(data_root),
        build_folador_manifest(data_root),
        build_hemato_manifest(data_root),
    ]
    df = pd.concat(parts, ignore_index=True)
    return enforce_schema(df)
