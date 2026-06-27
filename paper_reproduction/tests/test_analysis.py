"""End-to-end check that the shipped results reproduce the paper's headline numbers.

This is the V0 gate: load results/ and confirm a few anchor values match the
manuscript exactly (catalog == shipped, off-diagonal means, the DINOv2 Hemato
ceiling). If these drift, the shipped predictions no longer back the paper.
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
        pytest.skip("shipped results/ not present")
    return load_results(RESULTS)


def test_catalog_matches_shipped_results(results):
    cells, _ = results
    assert {c.cell_id for c in all_cells()} == set(cells["cell_id"])


@pytest.mark.parametrize(
    "pipeline,recipe,expected",
    [
        ("dinov2_probe", "AUG-None", 0.604),
        ("huang_vgg19", "AUG-Huang", 0.627),
        ("dogan_triple", "AUG-Dogan", 0.565),
        ("folador_hog", "AUG-None", 0.537),
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
    assert s["accuracy"] == pytest.approx(0.739, abs=5e-4)
    assert s["auc"] == pytest.approx(0.752, abs=5e-4)
