from pathlib import Path

import pytest

from parkinson_model_train.data import (
    DatasetSplit,
    ImageRecord,
    labels_to_indices,
    load_dataset_records,
    load_manifest,
    split_by_subject,
    validate_image_paths,
    validate_split_labels,
    write_manifest,
)
from parkinson_model_train.data.datasets import load_hemato_records, load_image_tree_records, load_niats_folador_records


def test_load_manifest_resolves_relative_paths(tmp_path: Path) -> None:
    image_path = tmp_path / "subject-1.png"
    image_path.write_bytes(b"fake")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_path,subject_id,label\nsubject-1.png,s1,PD\n", encoding="utf-8")

    records = load_manifest(manifest)

    assert records == [ImageRecord(image_path=image_path, subject_id="s1", label="PD")]


def test_write_manifest_preserves_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake")
    manifest = tmp_path / "manifest.csv"
    records = [
        ImageRecord(
            image_path=image_path,
            subject_id="s1",
            label="PD",
            metadata={"dataset": "synthetic", "source_label": "PD"},
        )
    ]

    write_manifest(records, manifest)
    loaded_records = load_manifest(manifest)

    assert loaded_records == records


def test_write_manifest_supports_documented_csv(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake")
    manifest = tmp_path / "manifest.csv"
    records = [ImageRecord(image_path=image_path, subject_id="s1", label="PD", metadata={"dataset": "hemato"})]

    write_manifest(records, manifest)

    assert load_manifest(manifest) == records
    assert manifest.read_text(encoding="utf-8").splitlines()[0] == "image_path,subject_id,label,dataset"


def test_write_manifest_paths_are_relative_to_manifest_directory(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "subject-1.png"
    manifest = tmp_path / "manifests" / "manifest.csv"
    image_path.parent.mkdir()
    image_path.write_bytes(b"fake")

    write_manifest([ImageRecord(image_path=image_path, subject_id="s1", label="PD")], manifest)

    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert lines[1].startswith("../images/subject-1.png,s1,PD")
    assert load_manifest(manifest)[0].image_path == image_path


def test_load_manifest_requires_expected_columns(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_path,label\nsubject-1.png,PD\n", encoding="utf-8")

    with pytest.raises(ValueError, match="subject_id"):
        load_manifest(manifest)


def test_load_manifest_rejects_non_csv_files(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match=".csv"):
        load_manifest(manifest)


def test_validate_image_paths_reports_missing_files(tmp_path: Path) -> None:
    records = [ImageRecord(image_path=tmp_path / "missing.png", subject_id="s1", label="PD")]

    with pytest.raises(ValueError, match="Missing image files"):
        validate_image_paths(records)


def test_split_by_subject_prevents_subject_leakage() -> None:
    records = [
        ImageRecord(
            image_path=Path(f"/tmp/{subject}-{index}.png"),
            subject_id=subject,
            label="PD" if subject in {"s1", "s2", "s3"} else "HC",
        )
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


def test_split_by_subject_rejects_conflicting_subject_labels() -> None:
    records = [
        ImageRecord(image_path=Path("pd.png"), subject_id="s1", label="PD"),
        ImageRecord(image_path=Path("hc.png"), subject_id="s1", label="HC"),
        ImageRecord(image_path=Path("pd2.png"), subject_id="s2", label="PD"),
        ImageRecord(image_path=Path("hc2.png"), subject_id="s3", label="HC"),
    ]

    with pytest.raises(ValueError, match="conflicting labels"):
        split_by_subject(records, val_fraction=0.2, test_fraction=0.0, seed=7)


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


def test_validate_split_labels_requires_eval_splits_to_include_training_labels() -> None:
    split = DatasetSplit(
        train=[
            ImageRecord(image_path=Path("pd-1.png"), subject_id="s1", label="PD"),
            ImageRecord(image_path=Path("hc-1.png"), subject_id="s2", label="HC"),
        ],
        val=[ImageRecord(image_path=Path("pd-2.png"), subject_id="s3", label="PD")],
        test=[],
    )

    with pytest.raises(ValueError, match="validation: HC"):
        validate_split_labels(split)


def test_load_hemato_records_reads_only_final_crops(tmp_path: Path) -> None:
    final_image = tmp_path / "V13" / "V13_PARKINSON_RAW_DATA_drawings_9" / "5_final" / "test1_spi_centered.png"
    ignored_image = tmp_path / "V13" / "V13_PARKINSON_RAW_DATA_drawings_9" / "2_cropping" / "test1_spi.png"
    final_image.parent.mkdir(parents=True)
    ignored_image.parent.mkdir(parents=True)
    final_image.write_bytes(b"fake")
    ignored_image.write_bytes(b"fake")

    records = load_hemato_records(tmp_path)

    assert records == [
        ImageRecord(
            image_path=final_image,
            subject_id="V13",
            label="PD",
            metadata={
                "dataset": "hemato",
                "drawing_type": "spiral",
                "drawing_session": "V13_PARKINSON_RAW_DATA_drawings_9",
                "source_label": "PD",
                "label_source": "filename",
            },
        )
    ]


def test_load_image_tree_records_reads_generic_labelled_folders(tmp_path: Path) -> None:
    patient_image = tmp_path / "parkinson" / "subject-pd" / "drawing_spi.png"
    control_image = tmp_path / "healthy" / "subject-hc" / "drawing_wave.png"
    patient_image.parent.mkdir(parents=True)
    control_image.parent.mkdir(parents=True)
    patient_image.write_bytes(b"fake")
    control_image.write_bytes(b"fake")

    records = load_image_tree_records(tmp_path)

    assert {record.label for record in records} == {"HC", "PD"}
    assert {record.subject_id for record in records} == {"subject-hc", "subject-pd"}
    assert {record.metadata["dataset"] for record in records} == {"image-tree"}
    assert {record.metadata["drawing_type"] for record in records} == {"meander", "spiral"}


def test_load_niats_folador_records_reads_dataset_junto_layout(tmp_path: Path) -> None:
    patient_image = (
        tmp_path
        / "dataset-junto"
        / "espiral"
        / "treinamento"
        / "parkinson"
        / "1_2_PE3.jpg"
    )
    control_image = (
        tmp_path
        / "dataset-junto"
        / "ondas"
        / "teste"
        / "higido"
        / "2_5_HO1.jpg"
    )
    patient_image.parent.mkdir(parents=True)
    control_image.parent.mkdir(parents=True)
    patient_image.write_bytes(b"fake")
    control_image.write_bytes(b"fake")

    records = load_niats_folador_records(tmp_path)

    assert {(record.subject_id, record.label) for record in records} == {("PE-2", "PD"), ("HO-5", "HC")}
    assert {record.metadata["dataset"] for record in records} == {"niats-folador"}
    assert {record.metadata["drawing_type"] for record in records} == {"meander", "spiral"}
    assert {record.metadata["source_subset"] for record in records} == {"dataset-junto"}
    assert {record.metadata["split"] for record in records} == {"teste", "treinamento"}
    assert {record.metadata["collection_day"] for record in records} == {"1", "2"}


def test_load_niats_folador_records_reads_dataset_separado_layout(tmp_path: Path) -> None:
    image_path = (
        tmp_path
        / "dataset-separado"
        / "dia2"
        / "espiral"
        / "teste"
        / "parkinson"
        / "2_11_PE4.jpg"
    )
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake")

    records = load_niats_folador_records(tmp_path)

    assert records == [
        ImageRecord(
            image_path=image_path,
            subject_id="PE-11",
            label="PD",
            metadata={
                "dataset": "niats-folador",
                "drawing_type": "spiral",
                "collection_day": "2",
                "source_subset": "dataset-separado",
                "split": "teste",
                "source_code": "PE",
                "source_label": "PD",
                "label_source": "filename",
            },
        )
    ]


def test_load_dataset_records_rejects_unknown_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported dataset"):
        load_dataset_records("unknown", tmp_path)


def test_split_by_subject_is_stratified() -> None:
    # 10 subjects: 6 PD, 4 HC
    records = []
    for i in range(6):
        records.append(ImageRecord(image_path=Path(f"/tmp/pd_{i}.png"), subject_id=f"pd_{i}", label="PD"))
    for i in range(4):
        records.append(ImageRecord(image_path=Path(f"/tmp/hc_{i}.png"), subject_id=f"hc_{i}", label="HC"))

    split = split_by_subject(records, val_fraction=0.25, test_fraction=0.25, seed=42)

    # PD: 6 * 0.25 = 1.5 -> round to 2 in val, 2 in test, 2 in train
    # HC: 4 * 0.25 = 1.0 -> round to 1 in val, 1 in test, 2 in train
    val_labels = [r.label for r in split.val]
    test_labels = [r.label for r in split.test]
    train_labels = [r.label for r in split.train]

    assert val_labels.count("PD") == 2
    assert val_labels.count("HC") == 1
    assert test_labels.count("PD") == 2
    assert test_labels.count("HC") == 1
    assert train_labels.count("PD") == 2
    assert train_labels.count("HC") == 2
