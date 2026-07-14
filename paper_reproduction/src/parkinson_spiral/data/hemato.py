"""Hemato (NIATS longitudinal clinical dataset) manifest builder.

Hemato is IRB-restricted clinical data and is NOT shipped with this release. It
is available on request from NIATS-UFU (see docs/DATA.md). This builder turns
the dataset's on-disk output into a unified manifest.

Preferred layout — flat session structure (no central parquet):
    <data_root>/hemato/output/<session_dir>/5_final/*_centered.png

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

import hashlib
import re
import warnings
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


def _session_fingerprint(final_dir: Path, image_paths: list[Path]) -> frozenset[str]:
    """SHA256-hash every image in the session; return the frozenset of hashes.

    Used to detect sessions whose image content is byte-identical to a previously
    processed session for the same subject (e.g. the V5/V17 _22 duplicates).
    """
    return frozenset(hashlib.sha256(p.read_bytes()).hexdigest() for p in image_paths)


def _scan_flat_output(output_root: Path, dir_name: str) -> pd.DataFrame:
    """Build manifest from a flat session structure (no central parquet needed).

    Scans every session directory under output_root for 5_final/*_centered.png.
    Sessions whose image content is byte-identical to a previously seen session
    for the same subject are skipped and reported via a warning.

    dir_name is used in the logical relative path for stable image_id hashing
    and must match the directory name under data_root/hemato/.
    """
    rows: list[dict] = []
    # (subject_id, frozenset-of-sha256s) → first session_id that had this content.
    seen: dict[tuple[str, frozenset[str]], str] = {}

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

        image_paths = sorted(final_dir.glob("*_centered.png"))
        if not image_paths:
            continue

        fingerprint = _session_fingerprint(final_dir, image_paths)
        key = (subject_id, fingerprint)
        if key in seen:
            warnings.warn(
                f"Hemato: skipping session {session_id!r} — byte-identical to {seen[key]!r} for subject {subject_id}",
                stacklevel=2,
            )
            continue
        seen[key] = session_id

        for image_path in image_paths:
            drawing_type = _drawing_type_from_filename(image_path.name)
            if drawing_type is None:
                continue
            abs_path = image_path.resolve()
            # Use the logical path for image_id stability: if the output dir is a
            # symlink outside data_root, abs_path.resolve() points to the symlink
            # target and relative_to(data_root) would fail.
            logical_rel = (Path("hemato") / dir_name / session_dir.name / "5_final" / image_path.name).as_posix()
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

    Priority order:
    1. output/ without all_metadata.parquet → flat session scan (preferred).
    2. output/ with all_metadata.parquet    → legacy parquet scan.
    3. output_v2/                           → flat session scan (old symlink name).
    4. Neither present                      → empty DataFrame.
    """
    data_root = Path(data_root)

    output_dir = data_root / "hemato" / "output"
    if output_dir.is_dir():
        parquet_path = output_dir / "all_metadata.parquet"
        if parquet_path.is_file():
            return _scan_output_legacy(parquet_path, data_root)
        return _scan_flat_output(output_dir, "output")

    # Backward compat: old symlink name.
    v2_root = data_root / "hemato" / "output_v2"
    if v2_root.is_dir():
        return _scan_flat_output(v2_root, "output_v2")

    return pd.DataFrame([], columns=MANIFEST_COLUMNS)
