from pathlib import Path

import pytest

from parkinson_model_train.data import (
    DatasetSplit,
    ImageRecord,
    labels_to_indices,
    load_dataset_records,
    load_folador_records,
    load_handpd_records,
    load_hemato_records,
    load_kaggle_drawings_records,
    load_kaggle_spiral_records,
    load_manifest,
    load_newhandpd_records,
    load_roboflow_records,
    split_by_subject,
    validate_image_paths,
    validate_split_labels,
    write_manifest,
)


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


def test_write_manifest_paths_are_relative_to_manifest_directory(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "subject-1.png"
    manifest = tmp_path / "manifests" / "manifest.csv"
    image_path.parent.mkdir()
    image_path.write_bytes(b"fake")

    write_manifest([ImageRecord(image_path=image_path, subject_id="s1", label="PD")], manifest)

    assert manifest.read_text(encoding="utf-8").splitlines()[1].startswith("../images/subject-1.png,")
    assert load_manifest(manifest)[0].image_path == image_path


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


def test_load_handpd_records_infers_labels_and_metadata(tmp_path: Path) -> None:
    patient_image = tmp_path / "SpiralPatients" / "0001-1.jpg"
    control_image = tmp_path / "MeanderControl" / "0059-5.jpg"
    patient_image.parent.mkdir()
    control_image.parent.mkdir()
    patient_image.write_bytes(b"fake")
    control_image.write_bytes(b"fake")

    records = load_handpd_records(tmp_path)

    assert {record.label for record in records} == {"HC", "PD"}
    assert {record.subject_id for record in records} == {"0001", "0059"}
    assert {record.metadata["drawing_type"] for record in records} == {"meander", "spiral"}
    assert {record.metadata["label_source"] for record in records} == {"filename"}


def test_load_newhandpd_records_infers_subjects_from_filenames(tmp_path: Path) -> None:
    patient_image = tmp_path / "PatientSpiral" / "sp1-P1.jpg"
    healthy_image = tmp_path / "HealthyMeander" / "mea1-H2.jpg"
    patient_image.parent.mkdir()
    healthy_image.parent.mkdir()
    patient_image.write_bytes(b"fake")
    healthy_image.write_bytes(b"fake")

    records = load_newhandpd_records(tmp_path)

    assert {record.label for record in records} == {"HC", "PD"}
    assert {record.subject_id for record in records} == {"H2", "P1"}


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
            subject_id="V13_PARKINSON_RAW_DATA_drawings_9",
            label="PD",
            metadata={
                "dataset": "hemato",
                "drawing_type": "spiral",
                "source_label": "PD",
                "label_source": "filename",
            },
        )
    ]


def test_load_kaggle_spiral_records_uses_image_level_subjects(tmp_path: Path) -> None:
    healthy_image = tmp_path / "healthy ID_0" / "0_file_df_1.png"
    patient_image = tmp_path / "parkinson ID_0" / "0_file_df_1.png"
    healthy_image.parent.mkdir()
    patient_image.parent.mkdir()
    healthy_image.write_bytes(b"fake")
    patient_image.write_bytes(b"fake")

    records = load_kaggle_spiral_records(tmp_path)

    assert {record.label for record in records} == {"HC", "PD"}
    assert {record.subject_id for record in records} == {"hc-0_file_df_1", "pd-0_file_df_1"}
    assert {record.metadata["source_dir"] for record in records} == {"healthy ID_0", "parkinson ID_0"}


def test_load_kaggle_drawings_records_reads_spiral_wave_layout(tmp_path: Path) -> None:
    healthy_image = tmp_path / "spiral" / "training" / "healthy" / "V01HE01.png"
    patient_image = tmp_path / "wave" / "testing" / "parkinson" / "V03PE01.png"
    healthy_image.parent.mkdir(parents=True)
    patient_image.parent.mkdir(parents=True)
    healthy_image.write_bytes(b"fake")
    patient_image.write_bytes(b"fake")

    records = load_kaggle_drawings_records(tmp_path)

    assert {record.label for record in records} == {"HC", "PD"}
    assert {record.metadata["dataset"] for record in records} == {"kaggle-drawings"}
    assert {record.metadata["kaggle_slug"] for record in records} == {"kmader/parkinsons-drawings"}
    assert {record.metadata["drawing_type"] for record in records} == {"spiral", "meander"}
    assert {record.metadata["source_split"] for record in records} == {"train", "test"}
    assert {record.subject_id for record in records} == {
        "hc-spiral-spiral-training-healthy-v01he01",
        "pd-meander-wave-testing-parkinson-v03pe01",
    }


def test_load_roboflow_records_uses_original_stem_as_subject(tmp_path: Path) -> None:
    healthy_image = tmp_path / "train" / "spiral" / "healthy" / "Healthy1100_png.rf.abc.jpg"
    patient_image = tmp_path / "valid" / "wave" / "parkinson" / "Parkinson4_png.rf.def.jpg"
    healthy_image.parent.mkdir(parents=True)
    patient_image.parent.mkdir(parents=True)
    healthy_image.write_bytes(b"fake")
    patient_image.write_bytes(b"fake")

    records = load_roboflow_records(tmp_path)

    assert {record.label for record in records} == {"HC", "PD"}
    assert {record.subject_id for record in records} == {"Healthy1100", "Parkinson4"}
    assert {record.metadata["drawing_type"] for record in records} == {"meander", "spiral"}
    assert {record.metadata["source_split"] for record in records} == {"train", "valid"}


def test_load_folador_records_reads_dataset_junto_layout(tmp_path: Path) -> None:
    dataset_root = tmp_path / "HOG-IA-desenho-manuscrito" / "pyTremorDetect" / "dataset-junto"
    healthy_image = dataset_root / "espiral" / "treinamento" / "higido" / "1_10_HE085.jpg"
    patient_image = dataset_root / "ondas" / "teste" / "parkinson" / "2_3_PE033.jpg"
    healthy_image.parent.mkdir(parents=True)
    patient_image.parent.mkdir(parents=True)
    healthy_image.write_bytes(b"fake")
    patient_image.write_bytes(b"fake")

    records = load_folador_records(tmp_path)

    assert {record.label for record in records} == {"HC", "PD"}
    assert {record.subject_id for record in records} == {"HE085", "PE033"}
    assert {record.metadata["drawing_type"] for record in records} == {"meander", "spiral"}
    assert {record.metadata["source_split"] for record in records} == {"teste", "treinamento"}


def test_load_dataset_records_rejects_unknown_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported dataset"):
        load_dataset_records("unknown", tmp_path)
