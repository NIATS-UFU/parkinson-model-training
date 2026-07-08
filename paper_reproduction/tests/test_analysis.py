"""End-to-end check that local results reproduce the paper's headline numbers.

Load results/ when present and confirm a few anchor values match the manuscript
exactly (catalog == results, off-diagonal means, the DINOv2 Hemato ceiling). If
these drift, the generated predictions no longer back the paper.
"""

from pathlib import Path

import pytest
from parkinson_spiral.analysis import (
    load_results,
    offdiagonal_mean,
    subject_suite,
)
from parkinson_spiral.catalog import all_cells

RESULTS = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(scope="module")
def results():
    if not (RESULTS / "cells.parquet").is_file():
        pytest.skip("local results/ not present")
    return load_results(RESULTS)


def test_catalog_matches_local_results(results):
    cells, _ = results
    assert {c.cell_id for c in all_cells()} == set(cells["cell_id"])


@pytest.mark.parametrize(
    "pipeline,recipe,expected",
    [
        # Values reflect corrected kaggle subject IDs (28 subjects, not 130 pseudo-IDs),
        # StratifiedKFold on subjects, and 20 hemato duplicate removals. These differ
        # from paper v5 which used the uncorrected splits.
        ("dinov2_probe", "AUG-None", 0.5999),
        ("huang_vgg19", "AUG-Huang", 0.4591),
        ("dogan_triple", "AUG-Dogan", 0.5599),
        ("folador_hog", "AUG-None", 0.5401),
    ],
)
def test_offdiagonal_means(results, pipeline, recipe, expected):
    cells, preds = results
    assert offdiagonal_mean(cells, preds, pipeline, recipe) == pytest.approx(expected, abs=5e-4)


def test_dinov2_hemato_ceiling(results):
    cells, preds = results
    s = subject_suite(
        cells,
        preds,
        pipeline_name="dinov2_probe",
        recipe_name="AUG-None",
        train_source="hemato",
        test_source="hemato",
        split_source="cv-fold",
    )
    # Corrected from paper v5 (0.739/0.752) due to subject-level fold correction.
    assert s["accuracy"] == pytest.approx(0.7826, abs=5e-4)
    assert s["auc"] == pytest.approx(0.7797, abs=5e-4)
