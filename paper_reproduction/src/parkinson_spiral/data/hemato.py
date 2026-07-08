"""Hemato (NIATS longitudinal clinical dataset) manifest builder.

Hemato is IRB-restricted clinical data and is NOT shipped with this release. It
is available on request from NIATS-UFU (see docs/DATA.md). This builder is the
code that turns the dataset's metadata parquet into a unified manifest.

Source: <data_root>/hemato/output/all_metadata.parquet. Each row corresponds to
one cropped image at `final_centered_rel_path`. We derive subject_id,
class_label, drawing_type, and session_id.

Subject ID + class label parsing:
    The session folder leaf name has one of three patterns:
        V<n>_PARKINSON_RAW_DATA_drawings_<N>      → parkinson
        V<n>_PARKINSON_DADOS_DESENHOS_<N>         → parkinson
        V<n>_CONTROLE_RAW_DATA_drawings_<N>       → healthy
    subject_id is the V<n> prefix of the leaf folder.

Drawing type: row_type == "spi" → spiral, row_type == "sen" → wave.
Session ID: the leaf folder name (enables per-session aggregation).
"""

import re
from pathlib import Path

import pandas as pd

from .unified import MANIFEST_COLUMNS, make_image_id

# V<n>_<DATASET_TAG>_<...>_<N>
_LEAF_RE = re.compile(r"^(V\d+)_(PARKINSON|CONTROLE)_", re.IGNORECASE)
_ROW_TYPE_TO_DRAWING = {"spi": "spiral", "sen": "wave"}


def _parse_leaf(sheet_rel_dir: str) -> tuple[str, str, str]:
    """Return (subject_id, class_label, session_id) from sheet_rel_dir."""
    leaf = sheet_rel_dir.rstrip("/").split("/")[-1]
    m = _LEAF_RE.match(leaf)
    if not m:
        raise ValueError(f"Hemato leaf folder did not match expected pattern: {leaf!r}")
    v_prefix, dataset_tag = m.group(1).upper(), m.group(2).upper()
    label = "parkinson" if dataset_tag == "PARKINSON" else "healthy"
    return f"hemato:{v_prefix}", label, leaf


def build_hemato_manifest(data_root: Path) -> pd.DataFrame:
    data_root = Path(data_root)
    parquet_path = data_root / "hemato" / "output" / "all_metadata.parquet"
    if not parquet_path.is_file():
        return pd.DataFrame([], columns=MANIFEST_COLUMNS)

    meta = pd.read_parquet(parquet_path)
    meta = meta[meta["final_centered_exists"].astype(bool)].copy()

    output_root = data_root / "hemato" / "output"
    rows: list[dict] = []
    for _, r in meta.iterrows():
        sheet = str(r["sheet_rel_dir"])
        try:
            subject_id, class_label, session_id = _parse_leaf(sheet)
        except ValueError:
            continue
        row_type = str(r["row_type"]).lower()
        drawing_type = _ROW_TYPE_TO_DRAWING.get(row_type)
        if drawing_type is None:
            continue
        # The parquet's final_centered_rel_path can be stale for nested sessions;
        # reconstruct from the leaf folder so the path matches the on-disk layout.
        recorded = Path(str(r["final_centered_rel_path"]))
        filename = recorded.name
        stage = recorded.parent.name  # "5_final"
        v_prefix = subject_id.split(":")[-1]
        reconstructed = output_root / v_prefix / session_id / stage / filename
        if reconstructed.is_file():
            abs_path = reconstructed.resolve()
        else:
            abs_path = (output_root / recorded).resolve()
        if not abs_path.is_file():
            continue
        manifest_rel = abs_path.relative_to(data_root.resolve()).as_posix()
        rows.append({
            "image_id": make_image_id("hemato", manifest_rel),
            "dataset": "hemato",
            "subject_id": subject_id,
            "drawing_type": drawing_type,
            "class_label": class_label,
            "published_split": None,  # Hemato has no published split; cv_fold assigned downstream
            "cv_fold": pd.NA,
            "session_id": session_id,
            "file_path": str(abs_path),
            "source_notes": None,
        })
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
