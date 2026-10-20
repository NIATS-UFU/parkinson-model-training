"""Tests for the reported statistics and the dataset distances.

The paired-comparison interval is checked against the worked example printed in its source
(Berrar and Lozano 2013, Table 2 and pp. 199-200); the rest against values derivable by hand.
"""

import numpy as np
import pandas as pd
import pytest
from parkinson_spiral import distances, evaluation
from parkinson_spiral.metrics import binary_metrics


def test_qh_interval_reproduces_the_published_worked_example() -> None:
    """Comparison (1,4): 21 against 3 discordant cases of 50 at Holm level 0.005 -> 0.31, upper 0.54.

    The source counts errors where this module counts the reference being right, so its counts
    enter transposed; only the sign depends on that.
    """
    result = evaluation.qh_interval(21, 3, 50, 0.005)

    assert abs(result["qh_point_shrunk"]) == pytest.approx(0.31, abs=0.005)
    assert max(abs(result["qh_lo"]), abs(result["qh_hi"])) == pytest.approx(0.54, abs=0.01)


def test_qh_interval_reproduces_the_published_small_count_example() -> None:
    """Comparison (1,2): 2 against 0 of 50, printed as [-0.03, 0.10] at its Holm level 0.05/3."""
    result = evaluation.qh_interval(2, 0, 50, 0.05 / 3)

    assert result["qh_lo"] == pytest.approx(-0.03, abs=0.01)
    assert result["qh_hi"] == pytest.approx(0.10, abs=0.01)


def test_qh_interval_sign_follows_which_classifier_errs_less() -> None:
    favours_reference = evaluation.qh_interval(8, 2, 40, 0.05)
    favours_competitor = evaluation.qh_interval(2, 8, 40, 0.05)

    assert favours_reference["qh_point_shrunk"] > 0
    assert favours_competitor["qh_lo"] == pytest.approx(-favours_reference["qh_hi"])
    assert favours_competitor["qh_hi"] == pytest.approx(-favours_reference["qh_lo"])


def test_mcnemar_exact_counts_discordant_participants() -> None:
    reference = np.array([1, 1, 1, 1, 1, 0, 0])
    competitor = np.array([0, 0, 0, 0, 0, 0, 1])

    result = evaluation.mcnemar_exact(reference, competitor)

    assert (result["discordant_b"], result["discordant_c"]) == (5, 1)
    assert result["p_mcnemar_exact"] == pytest.approx(2 * (1 + 6) / 2**6)


def test_mcnemar_exact_with_identical_decisions_is_one() -> None:
    decisions = np.array([1, 0, 1, 1])

    assert evaluation.mcnemar_exact(decisions, decisions)["p_mcnemar_exact"] == 1.0


def test_holm_adjusts_step_down_and_keeps_input_order() -> None:
    adjusted, levels = evaluation.holm_levels([0.01, 0.04, 0.03])

    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert levels == pytest.approx([0.05 / 3, 0.05, 0.05 / 2])


def _subjects(y_true: list[int], y_proba: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"y_true": y_true, "y_proba_pos": y_proba})


def test_cell_intervals_points_equal_the_reported_metrics() -> None:
    rng = np.random.default_rng(3)
    labels = [1, 0] * 10
    subjects = _subjects(labels, list(np.clip(0.5 * np.array(labels) + rng.uniform(0, 0.6, 20), 0, 1)))

    row = evaluation.cell_intervals(subjects, seed=7)
    reported = binary_metrics(subjects.y_true.to_numpy(), subjects.y_proba_pos.to_numpy())

    assert row["n_one_class_resamples"] == 0
    assert row["points_equal_binary_metrics"] is True
    for metric in evaluation.METRICS:
        assert row[metric] == reported[metric]
        assert row[f"{metric}_lo"] <= row[metric] <= row[f"{metric}_hi"]


def test_cell_intervals_count_resamples_that_hold_one_class() -> None:
    """With one patient among four, many resamples miss the patient; they must be counted."""
    subjects = _subjects([1, 0, 0, 0], [0.9, 0.2, 0.3, 0.1])

    assert evaluation.cell_intervals(subjects, seed=1)["n_one_class_resamples"] > 0


def test_load_cells_refuses_a_run_with_two_seeds(tmp_path) -> None:
    pd.DataFrame({"cell_id": ["a", "b"], "seed": [1, 2]}).to_parquet(tmp_path / "cells.parquet")

    with pytest.raises(ValueError, match="holds 2 seeds"):
        evaluation.load_cells(tmp_path)


def test_sliced_distance_is_zero_to_itself_and_symmetric() -> None:
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(20, 5)), rng.normal(size=(20, 5)) + 1.0
    directions = distances.unit_directions(5, 64, rng)

    assert distances.sliced_w2_squared(x, x.copy(), directions) == 0.0
    assert distances.sliced_w2_squared(x, y, directions) == pytest.approx(distances.sliced_w2_squared(y, x, directions))


def test_sliced_distance_of_a_shifted_cloud_is_the_mean_squared_projected_shift() -> None:
    """Shifting a cloud by t shifts every projected quantile by theta . t, so each direction adds (theta . t)^2."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 4))
    shift = np.array([1.0, -2.0, 0.5, 0.0])
    directions = distances.unit_directions(4, 50, rng)

    assert distances.sliced_w2_squared(x, x + shift, directions) == pytest.approx(np.mean((directions @ shift) ** 2))


def _embeddings(offsets: dict[str, float], per_dataset: int = 30, images: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    rows = []
    for dataset, offset in offsets.items():
        for subject in range(per_dataset):
            centre = rng.normal(size=6) + offset
            for image in range(images):
                rows.append({
                    "dataset": dataset,
                    "subject_id": f"{dataset}-{subject:02d}",
                    "image_id": f"{dataset}-{subject:02d}-{image}",
                    "drawing_type": "spiral",
                    "class_label": "parkinson" if subject % 2 else "healthy",
                    **{f"f{k}": value for k, value in enumerate(centre + 0.1 * rng.normal(size=6))},
                })
    return pd.DataFrame(rows)


def test_participant_clouds_average_each_participants_drawings() -> None:
    embeddings = _embeddings({"kaggle": 0.0, "folador": 0.0, "hemato": 0.0})

    clouds = distances.participant_clouds(embeddings)

    first = embeddings[embeddings.subject_id == "kaggle-00"][[f"f{k}" for k in range(6)]].mean().to_numpy()
    assert {name: len(cloud) for name, cloud in clouds.items()} == {"kaggle": 30, "folador": 30, "hemato": 30}
    np.testing.assert_allclose(clouds["kaggle"][0], first)


def test_participant_clouds_need_all_three_datasets() -> None:
    embeddings = _embeddings({"kaggle": 0.0, "folador": 0.0})

    with pytest.raises(ValueError, match="no spiral embeddings for \\['hemato'\\]"):
        distances.participant_clouds(embeddings)


def test_a_separated_dataset_clears_its_floor_and_the_null() -> None:
    clouds = distances.participant_clouds(_embeddings({"kaggle": 0.0, "folador": 0.0, "hemato": 3.0}))

    table = distances.dataset_distances(clouds, n_directions=100, n_draws=50, n_per_sample=10)

    between = table[table.kind == "between"].set_index(["a", "b"])
    assert between.loc[("kaggle", "hemato"), "distance"] > between.loc[("kaggle", "folador"), "distance"]
    assert bool(between.loc[("kaggle", "hemato"), "exceeds_floor_p95"])
    assert bool(between.loc[("kaggle", "hemato"), "exceeds_null_p95"])
    assert set(table[table.kind == "within"].a) == set(distances.DATASETS)


def test_dataset_distances_refuse_a_dataset_too_small_for_two_halves() -> None:
    clouds = distances.participant_clouds(_embeddings({"kaggle": 0.0, "folador": 0.0, "hemato": 0.0}, per_dataset=12))

    with pytest.raises(ValueError, match="cannot give two halves of 10"):
        distances.dataset_distances(clouds, n_directions=10, n_draws=2, n_per_sample=10)
