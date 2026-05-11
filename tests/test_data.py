from pathlib import Path

import pytest

from parkinson_model_train.data import (
    DatasetSplit,
    ImageRecord,
    labels_to_indices,
    load_manifest,
    split_by_subject,
    validate_image_paths,
    validate_split_labels,
)


def test_load_manifest_resolves_relative_paths(tmp_path: Path) -> None:
    image_path = tmp_path / "subject-1.png"
    image_path.write_bytes(b"fake")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_path,subject_id,label\nsubject-1.png,s1,PD\n", encoding="utf-8")

    records = load_manifest(manifest)

    assert records == [ImageRecord(image_path=image_path, subject_id="s1", label="PD")]


def test_load_manifest_requires_expected_columns(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_path,label\nsubject-1.png,PD\n", encoding="utf-8")

    with pytest.raises(ValueError, match="subject_id"):
        load_manifest(manifest)


def test_validate_image_paths_reports_missing_files(tmp_path: Path) -> None:
    records = [ImageRecord(image_path=tmp_path / "missing.png", subject_id="s1", label="PD")]

    with pytest.raises(ValueError, match="Missing image files"):
        validate_image_paths(records)


def test_split_by_subject_prevents_subject_leakage() -> None:
    records = [
        ImageRecord(image_path=Path(f"/tmp/{subject}-{index}.png"), subject_id=subject, label="PD" if index else "HC")
        for subject in ["s1", "s2", "s3", "s4", "s5", "s6"]
        for index in [0, 1]
    ]

    split = split_by_subject(records, val_fraction=0.2, test_fraction=0.2, seed=7)

    train_subjects = {record.subject_id for record in split.train}
    val_subjects = {record.subject_id for record in split.val}
    test_subjects = {record.subject_id for record in split.test}
    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)
    assert len(split.train) + len(split.val) + len(split.test) == len(records)


def test_split_by_subject_uses_fractions_of_all_subjects() -> None:
    records = [
        ImageRecord(image_path=Path(f"/tmp/{subject}.png"), subject_id=subject, label="PD" if index % 2 else "HC")
        for index, subject in enumerate([f"s{subject_number}" for subject_number in range(20)])
    ]

    split = split_by_subject(records, val_fraction=0.2, test_fraction=0.2, seed=7)

    assert len({record.subject_id for record in split.val}) == 4
    assert len({record.subject_id for record in split.test}) == 4


def test_labels_to_indices_requires_two_classes() -> None:
    records = [ImageRecord(image_path=Path("image.png"), subject_id="s1", label="PD")]

    with pytest.raises(ValueError, match="At least two labels"):
        labels_to_indices(records)


def test_validate_split_labels_requires_two_training_classes() -> None:
    split = DatasetSplit(
        train=[ImageRecord(image_path=Path("pd.png"), subject_id="s1", label="PD")],
        val=[],
        test=[],
    )

    with pytest.raises(ValueError, match="Training split must contain at least two labels"):
        validate_split_labels(split)


def test_validate_split_labels_rejects_labels_absent_from_training() -> None:
    split = DatasetSplit(
        train=[
            ImageRecord(image_path=Path("pd-1.png"), subject_id="s1", label="PD"),
            ImageRecord(image_path=Path("hc-1.png"), subject_id="s2", label="HC"),
        ],
        val=[ImageRecord(image_path=Path("other.png"), subject_id="s3", label="OTHER")],
        test=[],
    )

    with pytest.raises(ValueError, match="validation: OTHER"):
        validate_split_labels(split)
