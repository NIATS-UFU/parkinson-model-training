"""Hemato manifest builder — the dataset the paper calls **Moura**.

The NIATS-UFU longitudinal clinical cohort, de-identified as the
``moura-neto-2023-2025`` release. IRB-restricted, so it is NOT shipped here;
available on request from NIATS-UFU.
Expected layout (see docs/DATA.md for counts and known duplicates):

    <data_root>/hemato/V<n>/{spiral,wave}/V<n>_<PARKINSON|CONTROL>_<SPIRAL|WAVE>_<drawing>_<sheet>.png

Filename convention:
    The SECOND number is the source sheet; the first is which drawing on it (every
    sheet carries exactly 5). subject_id comes from the ``V<n>`` folder, drawing_type
    from the subfolder, class_label from the filename token.

    session_id is ``V<n>_<drawing>``, so it groups by position on the sheet rather
    than by acquisition — the sheet is the acquisition unit. Reported metrics are
    subject-level, so nothing published depends on it; do not read session-level
    output as per-acquisition.
"""

import re
from pathlib import Path

import pandas as pd

from .unified import MANIFEST_COLUMNS, make_image_id

# V<n>_<PARKINSON|CONTROL>_<SPIRAL|WAVE>_<drawing>_<sheet>.png
# Captures <drawing> (position on the sheet); <sheet> is matched but unused.
_FILE_RE = re.compile(r"^(V\d+)_(PARKINSON|CONTROL)_[A-Z]+_(\d+)_\d+\.png$", re.IGNORECASE)
_LABEL = {"PARKINSON": "parkinson", "CONTROL": "healthy"}


def build_hemato_manifest(data_root: Path) -> pd.DataFrame:
    data_root = Path(data_root)
    hemato_root = data_root / "hemato"
    if not hemato_root.is_dir():
        return pd.DataFrame([], columns=MANIFEST_COLUMNS)

    rows: list[dict] = []
    for drawing_type in ("spiral", "wave"):
        for img in sorted(hemato_root.glob(f"V*/{drawing_type}/*.png")):
            m = _FILE_RE.match(img.name)
            if m is None:
                raise ValueError(f"Hemato image name did not match expected pattern: {img.name!r}")
            v_prefix, label_tag, group = m.group(1).upper(), m.group(2).upper(), m.group(3)
            subject_id = f"hemato:{v_prefix}"
            # image_id is keyed on the path relative to <data_root> (root-independent,
            # so it survives the data being moved or symlinked); file_path keeps the
            # resolved absolute path the image loader reads.
            manifest_rel = (Path("hemato") / img.relative_to(hemato_root)).as_posix()
            abs_path = img.resolve()
            rows.append({
                "image_id": make_image_id("hemato", manifest_rel),
                "dataset": "hemato",
                "subject_id": subject_id,
                "drawing_type": drawing_type,
                "class_label": _LABEL[label_tag],
                "published_split": None,  # Hemato has no published split; cv_fold assigned downstream
                "cv_fold": pd.NA,
                "session_id": f"{v_prefix}_{group}",
                "file_path": str(abs_path),
                "source_notes": "moura-neto-2023-2025 cleaned release",
            })
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
