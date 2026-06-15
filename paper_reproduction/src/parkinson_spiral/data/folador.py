"""Folador et al. 2021 dataset manifest builder.

Expected layout (place the public dataset under <data_root>/folador/):

    <data_root>/folador/HOG-IA-desenho-manuscrito/pyTremorDetect/dataset-junto/
        espiral/{treinamento,teste}/{higido,parkinson}/*.jpg
        ondas/{treinamento,teste}/{higido,parkinson}/*.jpg

Subject ID convention:
    Filename pattern `<day>_<pid>_<HE|PE|HO|PO><NN>.jpg`. The pid (1..20) is
    local to each class folder, so we fold the class letter into subject_id:
    `folador:1H` vs `folador:1P`. The label LETTER (H/P) is read from the
    filename prefix and cross-checked against the folder name (higido/parkinson).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .unified import MANIFEST_COLUMNS, make_image_id

# <day>_<pid>_<LABEL><NN>.jpg  where LABEL ∈ {HE,PE,HO,PO}
_FNAME_RE = re.compile(r"^(\d+)_(\d+)_([HP])([EO])\d+\.jpg$", re.IGNORECASE)

_FOLDER_TO_LABEL = {"higido": "healthy", "parkinson": "parkinson"}
_DRAWING_FOLDER_TO_TYPE = {"espiral": "spiral", "ondas": "wave"}


def build_folador_manifest(data_root: Path) -> pd.DataFrame:
    data_root = Path(data_root)
    rows: list[dict] = []
    junto = data_root / "folador" / "HOG-IA-desenho-manuscrito" / "pyTremorDetect" / "dataset-junto"
    if not junto.is_dir():
        return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)

    for drawing_folder, drawing_type in _DRAWING_FOLDER_TO_TYPE.items():
        for split_folder, split_label in (("treinamento", "train"), ("teste", "test")):
            for class_folder, class_label in _FOLDER_TO_LABEL.items():
                d = junto / drawing_folder / split_folder / class_folder
                if not d.is_dir():
                    continue
                for img_path in sorted(d.iterdir()):
                    if not img_path.is_file() or img_path.suffix.lower() != ".jpg":
                        continue
                    m = _FNAME_RE.match(img_path.name)
                    if m:
                        day, pid, hp_letter, _eo_letter = m.groups()
                        # H/P comes from the filename prefix; cross-check with the folder.
                        fname_label = "healthy" if hp_letter.upper() == "H" else "parkinson"
                        notes = (
                            None
                            if fname_label == class_label
                            else f"folador-prefix-{hp_letter}-mismatches-folder-{class_folder}"
                        )
                        subject_id = f"folador:{int(pid)}{hp_letter.upper()}"
                        session_id = f"day{day}"
                    else:
                        subject_id = f"folador:{img_path.stem}"
                        session_id = None
                        notes = "folador-filename-did-not-match-expected-pattern"

                    rel = img_path.relative_to(data_root).as_posix()
                    rows.append(
                        {
                            "image_id": make_image_id("folador", rel),
                            "dataset": "folador",
                            "subject_id": subject_id,
                            "drawing_type": drawing_type,
                            "class_label": class_label,
                            "published_split": split_label,
                            "cv_fold": pd.NA,
                            "session_id": session_id,
                            "file_path": str(img_path.resolve()),
                            "source_notes": notes,
                        }
                    )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
