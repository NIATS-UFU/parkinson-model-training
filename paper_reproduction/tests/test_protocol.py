"""CellSpec identity + validation contracts."""

import pytest
from parkinson_spiral import DEFAULT_SEED
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
