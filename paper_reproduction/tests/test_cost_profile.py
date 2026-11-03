"""Tests for the cost profile's cell selection and its row-to-entry reduction.

Both halves are pure, so the ways the profile can silently describe the wrong thing are
exercised here without training anything: picking a cell that is not in the catalog, and
reporting a cache lookup as if it were a fit.
"""

import pytest


def _row(**overrides) -> dict:
    row = {
        "cell_id": "abc123",
        "pipeline_name": "dinov2_probe",
        "recipe_name": "AUG-None",
        "train_source": "hemato",
        "test_source": "kaggle",
        "split_source": "cross-dataset",
        "n_train_images": 3487,
        "n_test_images": 102,
        "n_train_subjects": 46,
        "n_test_subjects": 28,
        "fit_seconds": 12.5,
        "fit_was_cached": False,
        "predict_seconds": 1.0,
        "infer_latency_ms_per_img": 9.8,
        "accelerator_device": "cuda",
        "gpu_baseline_mb": 0.0,
        "gpu_peak_mb": 647.0,
        "gpu_peak_delta_mb": 647.0,
        "gpu_peak_reserved_mb": 834.0,
        "host_rss_peak_mb": 1496.8,
        "subject_accuracy": 0.5,
        "subject_auc": 0.5,
        "python_version": "3.13.15",
        "platform": "Linux",
    }
    row.update(overrides)
    return row


def test_one_cell_per_pipeline_and_all_are_real_catalog_cells(paper) -> None:
    """The profile must describe configurations the paper actually runs."""
    specs = paper.cost_profile_specs()
    catalog = {cell.cell_id for cell in paper.all_cells()}

    assert [spec.pipeline_name for spec in specs] == [name for name, _, _ in paper.NATIVE]
    assert all(spec.cell_id in catalog for spec in specs)
    for spec in specs:
        assert spec.train_source == paper.COST_PROFILE_TRAIN
        assert spec.test_source == paper.COST_PROFILE_TEST
        assert spec.split_source == paper.COST_PROFILE_SPLIT


def test_each_pipeline_is_profiled_in_its_native_recipe(paper) -> None:
    """A pipeline measured under someone else's recipe would not be its deployment cost."""
    native = {pipeline: recipe for pipeline, recipe, _ in paper.NATIVE}
    for spec in paper.cost_profile_specs():
        assert spec.recipe_name == native[spec.pipeline_name]


def test_entry_carries_the_cost_fields_and_the_free_parameter_count(paper) -> None:
    """The H4 cost argument counts free hyperparameters, so the profile has to carry them."""
    pipeline_class = paper.get_pipeline_class("dinov2_probe")

    entry = paper.build_cost_profile_entry(_row(), pipeline_class)

    assert set(paper.COST_PROFILE_FIELDS) <= set(entry)
    assert entry["free_hyperparameters"] == list(pipeline_class.FREE_HYPERPARAMETERS)
    assert entry["n_free_hyperparameters"] == len(pipeline_class.FREE_HYPERPARAMETERS)
    assert entry["training_regime"] == paper.TRAINING_REGIME["dinov2_probe"]


def test_a_reused_fit_is_rejected(paper) -> None:
    """A cached fit costs milliseconds; reporting it as a training cost is the D3 defect."""
    with pytest.raises(ValueError, match="reused a cached fit"):
        paper.build_cost_profile_entry(_row(fit_was_cached=True), paper.get_pipeline_class("dinov2_probe"))


def test_identical_predictions_are_reported_as_identical(paper, tmp_path) -> None:
    """The comparison is what turns the profile into a replicate of the run's own cells."""
    import pandas as pd

    frame = pd.DataFrame({
        "image_id": ["kaggle:0001", "kaggle:0002"],
        "y_proba_pos": [0.25, 0.75],
    })
    reference = tmp_path / "results" / "predictions"
    reference.mkdir(parents=True)
    frame.to_parquet(reference / "abc123.parquet", index=False)
    profiled = tmp_path / "work" / "abc123" / "predictions"
    profiled.mkdir(parents=True)
    frame.to_parquet(profiled / "abc123.parquet", index=False)

    result = paper.compare_cost_profile_predictions(
        [{"cell_id": "abc123", "pipeline_name": "dinov2_probe"}],
        tmp_path / "results",
        tmp_path / "work",
    )

    assert result[0]["compared"] is True
    assert result[0]["identical_predictions"] is True
    assert result[0]["max_abs_proba_delta"] == 0.0


def test_a_moved_probability_is_reported_not_swallowed(paper, tmp_path) -> None:
    """A profile that hid a divergence from the run would be worse than no comparison."""
    import pandas as pd

    reference = tmp_path / "results" / "predictions"
    reference.mkdir(parents=True)
    pd.DataFrame({"image_id": ["kaggle:0001"], "y_proba_pos": [0.25]}).to_parquet(
        reference / "abc123.parquet", index=False
    )
    profiled = tmp_path / "work" / "abc123" / "predictions"
    profiled.mkdir(parents=True)
    pd.DataFrame({"image_id": ["kaggle:0001"], "y_proba_pos": [0.30]}).to_parquet(
        profiled / "abc123.parquet", index=False
    )

    result = paper.compare_cost_profile_predictions(
        [{"cell_id": "abc123", "pipeline_name": "dinov2_probe"}],
        tmp_path / "results",
        tmp_path / "work",
    )

    assert result[0]["identical_predictions"] is False
    assert result[0]["max_abs_proba_delta"] == pytest.approx(0.05)


def test_a_missing_reference_run_is_reported_as_not_compared(paper, tmp_path) -> None:
    """Absence of a reference must not read as agreement with it."""
    result = paper.compare_cost_profile_predictions(
        [{"cell_id": "abc123", "pipeline_name": "dinov2_probe"}],
        tmp_path / "results",
        tmp_path / "work",
    )

    assert result[0]["compared"] is False
    assert "identical_predictions" not in result[0]


def test_different_test_images_are_summarised_without_a_delta(paper) -> None:
    """With different test images there is no delta to print, and that must not crash the summary."""
    record = {
        "cell_id": "abc123",
        "pipeline_name": "dinov2_probe",
        "compared": True,
        "same_test_images": False,
        "max_abs_proba_delta": None,
        "identical_predictions": False,
    }

    line = paper.format_comparison_line(record)

    assert "DIFFERS" in line
    assert "test images are not the completed run's" in line


def test_a_compared_cell_is_summarised_with_its_delta(paper) -> None:
    """The usual case prints the verdict and the largest probability difference."""
    record = {
        "cell_id": "abc123",
        "pipeline_name": "dinov2_probe",
        "compared": True,
        "same_test_images": True,
        "max_abs_proba_delta": 0.0,
        "identical_predictions": True,
    }

    assert paper.format_comparison_line(record) == "  dinov2_probe   identical  max |proba delta| = 0.000e+00"
