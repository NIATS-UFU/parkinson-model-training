"""Hemato manifest builder contracts — release image tree and cropper output."""

from pathlib import Path

import pandas as pd
import pytest
from parkinson_spiral.data.hemato import build_hemato_manifest
from parkinson_spiral.data.unified import MANIFEST_COLUMNS

PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000a49444154789c6360000002000100ffff03000006"
    "000557bfabd40000000049454e44ae426082"
)


def _release_tree(root: Path, files: dict[str, list[str]]) -> Path:
    """Materialise <root>/hemato/V<n>/<drawing_type>/<name>.png for each given name."""
    for rel, names in files.items():
        subject, drawing_type = rel.split("/")
        d = root / "hemato" / subject / drawing_type
        d.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(names):
            # Vary the bytes so unrelated images are not byte-identical.
            (d / name).write_bytes(PNG_1X1 + bytes([i]))
    return root


def _cropper_session(root: Path, session: str, names: list[str]) -> Path:
    """Materialise <root>/hemato/output/<session>/5_final/<name>.png for each name."""
    d = root / "hemato" / "output" / session / "5_final"
    d.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        (d / name).write_bytes(PNG_1X1 + bytes([i]))
    return root


def test_release_tree_builds_expected_fields_from_the_path(tmp_path):
    _release_tree(tmp_path, {"V1/spiral": ["V1_PARKINSON_SPIRAL_1_7.png"], "V3/wave": ["V3_CONTROL_WAVE_2_4.png"]})

    df = build_hemato_manifest(tmp_path)

    assert list(df.columns) == list(MANIFEST_COLUMNS)
    by_subject = df.set_index("subject_id")
    assert by_subject.loc["hemato:V1", "class_label"] == "parkinson"
    assert by_subject.loc["hemato:V1", "drawing_type"] == "spiral"
    assert by_subject.loc["hemato:V3", "class_label"] == "healthy"
    assert by_subject.loc["hemato:V3", "drawing_type"] == "wave"
    assert by_subject.loc["hemato:V1", "published_split"] is None
    assert pd.isna(by_subject.loc["hemato:V1", "cv_fold"])


def test_release_session_id_is_the_drawing_index_not_the_sheet(tmp_path):
    """session_id groups by position on the sheet, not by acquisition.

    Drawing 2 of sheet 4 and drawing 2 of sheet 9 share a session_id, while drawing 3
    of sheet 4 does not. Session-level output is therefore not per-acquisition.
    """
    _release_tree(
        tmp_path,
        {
            "V1/spiral": [
                "V1_PARKINSON_SPIRAL_2_4.png",  # drawing 2, sheet 4
                "V1_PARKINSON_SPIRAL_2_9.png",  # drawing 2, sheet 9  -> same session_id
                "V1_PARKINSON_SPIRAL_3_4.png",  # drawing 3, sheet 4  -> different
            ]
        },
    )

    df = build_hemato_manifest(tmp_path)
    sessions = dict(zip(df.file_path.map(lambda p: Path(p).name), df.session_id, strict=True))

    assert sessions["V1_PARKINSON_SPIRAL_2_4.png"] == "V1_2"
    assert sessions["V1_PARKINSON_SPIRAL_2_9.png"] == "V1_2"
    assert sessions["V1_PARKINSON_SPIRAL_3_4.png"] == "V1_3"


def test_release_unparseable_filename_raises(tmp_path):
    _release_tree(tmp_path, {"V1/spiral": ["V1_PARKINSON_SPIRAL_1_1.png", "not-a-hemato-name.png"]})

    with pytest.raises(ValueError, match=r"Hemato image name did not match expected pattern: 'not-a-hemato-name\.png'"):
        build_hemato_manifest(tmp_path)


def test_release_image_id_is_stable_across_data_roots(tmp_path):
    """image_id keys on the data-root-relative path, so moving the tree must not change it."""
    a = _release_tree(tmp_path / "a", {"V1/spiral": ["V1_PARKINSON_SPIRAL_1_1.png"]})
    b = _release_tree(tmp_path / "b", {"V1/spiral": ["V1_PARKINSON_SPIRAL_1_1.png"]})

    assert build_hemato_manifest(a).loc[0, "image_id"] == build_hemato_manifest(b).loc[0, "image_id"]


def test_missing_dataset_yields_empty_manifest(tmp_path):
    """An absent dataset is skipped, so a Kaggle+Folador-only root still builds."""
    df = build_hemato_manifest(tmp_path)

    assert df.empty
    assert list(df.columns) == list(MANIFEST_COLUMNS)


def test_cropper_output_layout_still_loads(tmp_path):
    """The cropper flat layout (output/<session>/5_final/*_centered.png) is unchanged."""
    _cropper_session(tmp_path, "V1_PARKINSON_RAW_DATA_drawings_1", ["test1_spi_centered.png", "test1_sen_centered.png"])

    df = build_hemato_manifest(tmp_path)

    assert df["subject_id"].unique().tolist() == ["hemato:V1"]
    assert df["class_label"].unique().tolist() == ["parkinson"]
    assert set(df["drawing_type"]) == {"spiral", "wave"}


def test_cropper_output_takes_precedence_over_release_tree(tmp_path):
    """When both layouts are present, the cropper output wins (release is a fallback)."""
    _cropper_session(tmp_path, "V1_PARKINSON_RAW_DATA_drawings_1", ["test1_spi_centered.png"])
    _release_tree(tmp_path, {"V1/spiral": ["V1_PARKINSON_SPIRAL_1_1.png"]})

    df = build_hemato_manifest(tmp_path)

    # The cropper flat scan leaves source_notes empty; the release scan stamps a note.
    assert df["source_notes"].dropna().empty
    assert len(df) == 1
