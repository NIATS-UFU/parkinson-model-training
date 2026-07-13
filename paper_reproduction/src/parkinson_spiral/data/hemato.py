"""Hemato (NIATS longitudinal clinical dataset) manifest builder.

Hemato is IRB-restricted clinical data and is NOT shipped with this release. It
is available on request from NIATS-UFU (see docs/DATA.md). This builder turns
the dataset's on-disk output into a unified manifest.

Preferred layout — output_v2 (flat session structure, no central parquet):
    <data_root>/hemato/output_v2/<session_dir>/5_final/*_centered.png

where each <session_dir> matches one of:
    V<n>_PARKINSON_RAW_DATA_drawings_<N>   → parkinson
    V<n>_PARKINSON_DADOS_DESENHOS_<N>      → parkinson
    V<n>_CONTROLE_RAW_DATA_drawings_<N>    → healthy

Legacy layout (output/ with central parquet, kept as fallback):
    <data_root>/hemato/output/all_metadata.parquet

Subject ID + class label parsing:
    subject_id is the V<n> prefix of the session folder name.
    drawing type is inferred from the filename: *_spi_* → spiral, *_sen_* → wave.
"""

import re
from pathlib import Path

import pandas as pd

from .unified import MANIFEST_COLUMNS, make_image_id

# V<n>_<DATASET_TAG>_<...>
_LEAF_RE = re.compile(r"^(V\d+)_(PARKINSON|CONTROLE)_", re.IGNORECASE)
_ROW_TYPE_TO_DRAWING = {"spi": "spiral", "sen": "wave"}


def _parse_leaf(leaf: str) -> tuple[str, str, str]:
    """Return (subject_id, class_label, session_id) from a session folder name."""
    m = _LEAF_RE.match(leaf.rstrip("/").split("/")[-1])
    if not m:
        raise ValueError(f"Hemato leaf folder did not match expected pattern: {leaf!r}")
    v_prefix, dataset_tag = m.group(1).upper(), m.group(2).upper()
    label = "parkinson" if dataset_tag == "PARKINSON" else "healthy"
    return f"hemato:{v_prefix}", label, leaf.rstrip("/").split("/")[-1]


def _drawing_type_from_filename(name: str) -> str | None:
    """Return 'spiral' or 'wave' from *_spi_* / *_sen_* filenames, else None."""
    lower = name.lower()
    if "_spi_" in lower:
        return "spiral"
    if "_sen_" in lower:
        return "wave"
    return None


def _scan_output_v2(output_root: Path, data_root: Path) -> pd.DataFrame:
    """Build manifest from output_v2 flat session structure (no parquet needed).

    Scans every session directory under output_root for 5_final/*_centered.png
    and records the absolute path, subject identity, and drawing type.
    """
    rows: list[dict] = []
    for session_dir in sorted(output_root.iterdir()):
        if not session_dir.is_dir():
            continue
        try:
            subject_id, class_label, session_id = _parse_leaf(session_dir.name)
        except ValueError:
            continue
        final_dir = session_dir / "5_final"
        if not final_dir.is_dir():
            continue
        for image_path in sorted(final_dir.glob("*_centered.png")):
            drawing_type = _drawing_type_from_filename(image_path.name)
            if drawing_type is None:
                continue
            abs_path = image_path.resolve()
            # Use the logical path through data_root for image_id stability: if output_v2 is a
            # symlink outside data_root, abs_path.resolve() would point to the symlink target
            # and relative_to(data_root) would fail. The logical path is always stable.
            logical_rel = (Path("hemato") / "output_v2" / session_dir.name / "5_final" / image_path.name).as_posix()
            rows.append({
                "image_id": make_image_id("hemato", logical_rel),
                "dataset": "hemato",
                "subject_id": subject_id,
                "drawing_type": drawing_type,
                "class_label": class_label,
                "published_split": None,
                "cv_fold": pd.NA,
                "session_id": session_id,
                "file_path": str(abs_path),
                "source_notes": None,
            })
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def _scan_output_legacy(parquet_path: Path, data_root: Path) -> pd.DataFrame:
    """Build manifest from the legacy all_metadata.parquet + V<n>/session/ nesting."""
    meta = pd.read_parquet(parquet_path)
    meta = meta[meta["final_centered_exists"].astype(bool)].copy()

    output_root = parquet_path.parent
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
            "published_split": None,
            "cv_fold": pd.NA,
            "session_id": session_id,
            "file_path": str(abs_path),
            "source_notes": None,
        })
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def build_hemato_manifest(data_root: Path) -> pd.DataFrame:
    """Build the Hemato portion of the unified manifest.

    Checks for output_v2 first (preferred flat structure), then falls back to
    the legacy output/ directory with all_metadata.parquet.
    """
    data_root = Path(data_root)

    # Preferred: output_v2 flat session structure, no central parquet needed.
    v2_root = data_root / "hemato" / "output_v2"
    if v2_root.is_dir():
        return _scan_output_v2(v2_root, data_root)

    # Legacy fallback: output/ with central all_metadata.parquet.
    parquet_path = data_root / "hemato" / "output" / "all_metadata.parquet"
    if not parquet_path.is_file():
        return pd.DataFrame([], columns=MANIFEST_COLUMNS)
    return _scan_output_legacy(parquet_path, data_root)
