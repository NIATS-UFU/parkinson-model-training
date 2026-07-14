"""Hemato (NIATS longitudinal clinical dataset) manifest builder.

Hemato is IRB-restricted clinical data and is NOT shipped with this release. It
is available on request from NIATS-UFU (see docs/DATA.md). This builder turns the
cleaned on-disk image tree into a unified manifest.

Source layout — the cleaned ``moura-neto-2023-2025`` release placed under
``<data_root>/hemato/``::

    <data_root>/hemato/V<n>/spiral/V<n>_<PARKINSON|CONTROL>_SPIRAL_<group>_<idx>.png
    <data_root>/hemato/V<n>/wave/V<n>_<PARKINSON|CONTROL>_WAVE_<group>_<idx>.png

Fields are derived from the path: subject_id from the ``V<n>`` folder,
drawing_type from the ``spiral``/``wave`` subfolder, class_label from the
``PARKINSON``/``CONTROL`` filename token.

Session note: the cleaned release re-batched images into per-subject ``<group>``
indices and no longer carries the original acquisition-session folders. We use
``V<n>_<group>`` as session_id so the internal session-level aggregation still
runs, but it no longer maps to the original ``drawings_N`` sessions. The reported
paper metrics are subject-level (see metrics.py), so this does not affect them.
"""

import re
from pathlib import Path

import pandas as pd

from .unified import MANIFEST_COLUMNS, make_image_id

# V<n>_<PARKINSON|CONTROL>_<SPIRAL|WAVE>_<group>_<idx>.png
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
