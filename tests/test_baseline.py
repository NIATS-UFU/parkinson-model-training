from pathlib import Path

from train_baseline import compute_fold_metrics, compute_roc_auc, stratified_group_k_fold

from parkinson_model_train.data import ImageRecord


def test_stratified_group_k_fold_prevents_leakage() -> None:
    # Set up synthetic records for 6 patients (3 PD, 3 HC) with 2 drawings each
    records = []
    # Patients V1, V2, V3 are PD
    for p in ["V1", "V2", "V3"]:
        records.append(
            ImageRecord(
                image_path=Path(f"data/hemato/{p}/{p}_drawings_1/5_final/test3_spi_centered.png"),
                subject_id=f"{p}_drawings_1",
                label="PD",
            )
        )
        records.append(
            ImageRecord(
                image_path=Path(f"data/hemato/{p}/{p}_drawings_2/5_final/test3_spi_centered.png"),
                subject_id=f"{p}_drawings_2",
                label="PD",
            )
        )
    # Patients V4, V5, V6 are HC
    for p in ["V4", "V5", "V6"]:
        records.append(
            ImageRecord(
                image_path=Path(f"data/hemato/{p}/{p}_drawings_1/5_final/test3_spi_centered.png"),
                subject_id=f"{p}_drawings_1",
                label="HC",
            )
        )
        records.append(
            ImageRecord(
                image_path=Path(f"data/hemato/{p}/{p}_drawings_2/5_final/test3_spi_centered.png"),
                subject_id=f"{p}_drawings_2",
                label="HC",
            )
        )

    # Split into 3 folds
    splits = stratified_group_k_fold(records, num_folds=3, seed=42)

    assert len(splits) == 3
    for fold in splits:
        train_records = fold["train"]
        val_records = fold["val"]

        # Ensure no overlap in drawings
        train_paths = {r.image_path for r in train_records}
        val_paths = {r.image_path for r in val_records}
        assert train_paths.isdisjoint(val_paths)

        # Ensure no patient leakage
        train_patients = {r.image_path.parent.parent.parent.name for r in train_records}
        val_patients = {r.image_path.parent.parent.parent.name for r in val_records}
        assert train_patients.isdisjoint(val_patients)

        # Every record is accounted for
        assert len(train_records) + len(val_records) == len(records)


def test_compute_roc_auc_dummy() -> None:
    # Perfect classifier
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.8, 0.9]
    assert compute_roc_auc(y_true, y_scores) == 1.0

    # Opposite classifier
    y_true = [0, 0, 1, 1]
    y_scores = [0.9, 0.8, 0.2, 0.1]
    assert compute_roc_auc(y_true, y_scores) == 0.0

    # Random-like classifier
    y_true = [0, 1]
    y_scores = [0.5, 0.5]
    assert compute_roc_auc(y_true, y_scores) == 0.5


def test_compute_fold_metrics() -> None:
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 1]  # 1 TP, 1 TN, 1 FN, 1 FP
    y_prob = [0.8, 0.2, 0.4, 0.6]

    metrics = compute_fold_metrics(y_true, y_pred, y_prob)

    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "sensitivity" in metrics
    assert "specificity" in metrics
    assert "roc_auc" in metrics

    assert metrics["accuracy"] == 0.5
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5
