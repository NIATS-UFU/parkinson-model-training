"""CellSpec identity + validation contracts."""

from pathlib import Path

import pandas as pd
import pytest
from parkinson_spiral import DEFAULT_SEED
from parkinson_spiral.data.hemato import build_hemato_manifest
from parkinson_spiral.data.kaggle import _subject_id
from parkinson_spiral.data.splits import assign_subject_folds
from parkinson_spiral.protocol import CellSpec


def _spec(**over):
    base = dict(
        pipeline_name="dinov2_probe",
        recipe_name="AUG-None",
        train_source="hemato",
        test_source="hemato",
        split_source="cv-fold",
        cv_fold=0,
        task_filter="spiral",
        target_column="class_label",
        augment_then_split=False,
        seed=DEFAULT_SEED,
    )
    base.update(over)
    return CellSpec(**base)


def test_cell_id_is_deterministic_and_16_hex():
    spec = _spec()
    cid = spec.cell_id
    assert cid == _spec().cell_id  # same params -> same id
    assert len(cid) == 16
    int(cid, 16)  # hex-decodable


def test_cell_id_changes_with_any_field():
    base = _spec().cell_id
    assert _spec(train_source="kaggle", test_source="kaggle").cell_id != base
    assert _spec(recipe_name="AUG-Huang").cell_id != base
    assert _spec(cv_fold=1).cell_id != base


def test_validate_rejects_unknown_split_source():
    with pytest.raises(ValueError):
        _spec(split_source="image-random").validate()


def test_validate_requires_class_label_target():
    with pytest.raises(ValueError):
        _spec(target_column="updrs").validate()


def test_cv_fold_must_be_within_dataset():
    with pytest.raises(ValueError):
        _spec(train_source="kaggle", test_source="hemato").validate()


def test_cross_dataset_must_be_off_diagonal():
    with pytest.raises(ValueError):
        _spec(split_source="cross-dataset", cv_fold=None, train_source="kaggle", test_source="kaggle").validate()


def test_valid_spec_passes():
    _spec().validate()  # must not raise


def test_kaggle_spiral_and_wave_files_share_subject_identity():
    assert _subject_id("V03PE02.png", "parkinson") == "kaggle:V03P"
    assert _subject_id("V03PO02.png", "parkinson") == "kaggle:V03P"
    assert _subject_id("V07HE01.png", "healthy") == "kaggle:V07H"
    assert _subject_id("V07HO01.png", "healthy") == "kaggle:V07H"


def test_hemato_v2_scan_parses_parkinson_session(tmp_path: Path) -> None:
    session = tmp_path / "hemato" / "output_v2" / "V3_PARKINSON_RAW_DATA_drawings_1" / "5_final"
    session.mkdir(parents=True)
    (session / "test1_spi_centered.png").write_bytes(b"fake")
    (session / "test1_sen_centered.png").write_bytes(b"fake")

    df = build_hemato_manifest(tmp_path)

    assert len(df) == 2
    assert set(df["subject_id"]) == {"hemato:V3"}
    assert set(df["class_label"]) == {"parkinson"}
    assert set(df["drawing_type"]) == {"spiral", "wave"}
    assert set(df["session_id"]) == {"V3_PARKINSON_RAW_DATA_drawings_1"}


def test_hemato_v2_scan_parses_control_session(tmp_path: Path) -> None:
    session = tmp_path / "hemato" / "output_v2" / "V5_CONTROLE_RAW_DATA_drawings_2" / "5_final"
    session.mkdir(parents=True)
    (session / "test1_spi_centered.png").write_bytes(b"fake")

    df = build_hemato_manifest(tmp_path)

    assert len(df) == 1
    assert df.iloc[0]["subject_id"] == "hemato:V5"
    assert df.iloc[0]["class_label"] == "healthy"
    assert df.iloc[0]["drawing_type"] == "spiral"


def test_hemato_v2_scan_skips_non_matching_folders(tmp_path: Path) -> None:
    v2 = tmp_path / "hemato" / "output_v2"
    good = v2 / "V1_PARKINSON_RAW_DATA_drawings_1" / "5_final"
    good.mkdir(parents=True)
    (good / "test1_spi_centered.png").write_bytes(b"fake")
    # Folder without the expected naming pattern — should be ignored.
    bad = v2 / "misc_notes"
    bad.mkdir()

    df = build_hemato_manifest(tmp_path)

    assert len(df) == 1
    assert df.iloc[0]["subject_id"] == "hemato:V1"


def test_hemato_falls_back_to_empty_when_neither_source_exists(tmp_path: Path) -> None:
    df = build_hemato_manifest(tmp_path)
    assert df.empty


def test_fold_assignment_gives_each_subject_exactly_one_fold():
    manifest = pd.DataFrame({
        "dataset": ["kaggle"] * 24,
        "subject_id": [f"s{i}" for i in range(10) for _ in range(2)] + ["s10"] * 4,
        "class_label": ["healthy"] * 10 + ["parkinson"] * 14,
    })
    folded = assign_subject_folds(manifest, datasets=("kaggle",), n_splits=2)
    assert folded.groupby("subject_id")["cv_fold"].nunique().eq(1).all()
    subject_folds = folded[["subject_id", "class_label", "cv_fold"]].drop_duplicates()
    assert sorted(subject_folds.groupby("cv_fold")["subject_id"].size().tolist()) == [5, 6]
