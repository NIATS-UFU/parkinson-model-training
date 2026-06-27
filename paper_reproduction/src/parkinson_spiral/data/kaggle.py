"""Kaggle (Zham 2017 "Parkinson's Drawings") manifest builder.

Expected layout (place the public dataset under <data_root>/kaggle/):

    <data_root>/kaggle/{spiral,wave}/{training,testing}/{healthy,parkinson}/*.png

Subject ID convention:
    Filename pattern `V<NN><P|H><E|O><NN>.png`, where E denotes spiral
    (espiral) and O denotes wave (onda). The same V-number appears in BOTH
    healthy/ and parkinson/ folders but refers to different individuals
    (class-local indexing). To make subject-level grouping safe, we fold the
    class letter into subject_id: `kaggle:V01H` vs `kaggle:V01P`.
"""

import re
from pathlib import Path

import pandas as pd

from .unified import MANIFEST_COLUMNS, make_image_id

# V<NN>[P|H][E|O]<NN>.png — class letter identifies the participant and the
# E/O letter identifies the drawing task. Both modalities must map to the same
# participant before subject-grouped folds are assigned.
_FNAME_RE = re.compile(r"^(V\d+)([PH])[EO]\d+\.(?:png|jpe?g)$", re.IGNORECASE)


def _subject_id(stem_with_ext: str, class_label: str) -> str:
    m = _FNAME_RE.match(stem_with_ext)
    if m:
        return f"kaggle:{m.group(1).upper()}{m.group(2).upper()}"
    # Fallback: filename-as-subject (class-local-safe because the filename is unique within the class folder).
    base = Path(stem_with_ext).stem
    return f"kaggle:{base}-{class_label[0].upper()}"


def build_kaggle_manifest(data_root: Path) -> pd.DataFrame:
    data_root = Path(data_root)
    rows: list[dict] = []
    kaggle_root = data_root / "kaggle"
    if not (kaggle_root / "spiral").is_dir() and (kaggle_root / "drawings").is_dir():
        kaggle_root /= "drawings"
    for drawing_type in ("spiral", "wave"):
        for split_folder, split_label in (("training", "train"), ("testing", "test")):
            for label_folder, label in (("healthy", "healthy"), ("parkinson", "parkinson")):
                d = kaggle_root / drawing_type / split_folder / label_folder
                if not d.is_dir():
                    continue
                for img_path in sorted(d.iterdir()):
                    if not img_path.is_file() or img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                        continue
                    rel = img_path.relative_to(data_root).as_posix()
                    sid = _subject_id(img_path.name, label)
                    rows.append({
                        "image_id": make_image_id("kaggle", rel),
                        "dataset": "kaggle",
                        "subject_id": sid,
                        "drawing_type": drawing_type,
                        "class_label": label,
                        "published_split": split_label,
                        "cv_fold": pd.NA,
                        "session_id": None,
                        "file_path": str(img_path.resolve()),
                        "source_notes": None,
                    })
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
