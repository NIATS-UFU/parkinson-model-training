"""Post-hoc analysis: pooled out-of-fold CV ceilings, cross-dataset matrices.

Every function here reads a local `results/` directory (cells.parquet +
predictions/) and never triggers training. All reported numbers are subject-level accuracy
(the paper's canonical level), pooling predictions across the cells that make up
a logical result (e.g. the 5 CV folds of a diagonal).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import DEFAULT_SEED
from .metrics import aggregate_to_subject, binary_metrics, percentile_bootstrap_ci

SOURCES = ("kaggle", "folador", "hemato")


def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cells, predictions_long) from a results directory."""
    results_dir = Path(results_dir)
    cells = pd.read_parquet(results_dir / "cells.parquet")
    preds_dir = results_dir / "predictions"
    parts = [pd.read_parquet(p) for p in sorted(preds_dir.glob("*.parquet"))]
    preds = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return cells, preds


def _accuracy_only(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float((y_true == (y_proba >= 0.5).astype(int)).mean())


def _select(cells: pd.DataFrame, **filters) -> pd.DataFrame:
    mask = pd.Series(True, index=cells.index)
    for col, val in filters.items():
        mask &= cells[col] == val
    return cells[mask]


def subject_accuracy(
    cells: pd.DataFrame,
    preds: pd.DataFrame,
    *,
    bootstrap: bool = True,
    seed: int = DEFAULT_SEED,
    **filters,
) -> dict[str, float] | None:
    """Pool predictions across all cells matching `filters`, aggregate to subject,
    return {point, ci_lo, ci_hi, n_cells, n_subjects}. None if no cells match.

    Pooling several CV-fold cells gives the pooled out-of-fold ceiling; a single
    cross-dataset / published cell just scores that cell at the subject level."""
    sub_cells = _select(cells, **filters)
    if sub_cells.empty:
        return None
    cell_ids = sub_cells["cell_id"].tolist()
    pdf = preds[preds["cell_id"].isin(cell_ids)]
    subj = aggregate_to_subject(pdf)
    if subj.empty:
        return None
    point = _accuracy_only(subj["y_true"].to_numpy(), subj["y_proba_pos"].to_numpy())
    out = {"point": float(point), "n_cells": len(cell_ids), "n_subjects": int(len(subj))}
    if bootstrap:
        ci = percentile_bootstrap_ci(subj, _accuracy_only, seed=seed)
        out["ci_lo"] = ci["ci_lo"]
        out["ci_hi"] = ci["ci_hi"]
    return out


def subject_suite(
    cells: pd.DataFrame,
    preds: pd.DataFrame,
    *,
    bootstrap: bool = True,
    seed: int = DEFAULT_SEED,
    **filters,
) -> dict[str, float] | None:
    """Full subject-level metric suite (accuracy, sensitivity, specificity, precision,
    f1, auc) over predictions pooled across all cells matching `filters`, plus the
    accuracy's percentile bootstrap 95% CI. None if no cells match."""
    sub_cells = _select(cells, **filters)
    if sub_cells.empty:
        return None
    pdf = preds[preds["cell_id"].isin(sub_cells["cell_id"].tolist())]
    subj = aggregate_to_subject(pdf)
    if subj.empty:
        return None
    out = dict(binary_metrics(subj["y_true"].to_numpy(), subj["y_proba_pos"].to_numpy()))
    out["n_subjects"] = int(len(subj))
    if bootstrap:
        ci = percentile_bootstrap_ci(subj, _accuracy_only, seed=seed)
        out["ci_lo"] = ci["ci_lo"]
        out["ci_hi"] = ci["ci_hi"]
    return out


def diagonal_ceiling(cells, preds, pipeline: str, recipe: str, source: str, **kw) -> dict | None:
    """Pooled out-of-fold subject-accuracy ceiling for one within-dataset diagonal."""
    return subject_accuracy(
        cells,
        preds,
        pipeline_name=pipeline,
        recipe_name=recipe,
        train_source=source,
        test_source=source,
        split_source="cv-fold",
        **kw,
    )


def cross_value(cells, preds, pipeline: str, recipe: str, train: str, test: str, **kw) -> dict | None:
    """Subject-accuracy for one cross-dataset off-diagonal transfer cell."""
    return subject_accuracy(
        cells,
        preds,
        pipeline_name=pipeline,
        recipe_name=recipe,
        train_source=train,
        test_source=test,
        split_source="cross-dataset",
        **kw,
    )


def cross_matrix(cells, preds, pipeline: str, recipe: str, bootstrap: bool = False) -> pd.DataFrame:
    """3x3 train x test subject-accuracy matrix for one pipeline. Diagonal = pooled
    CV ceiling; off-diagonal = cross-dataset transfer."""
    rows = []
    for train in SOURCES:
        row = {"train\\test": train}
        for test in SOURCES:
            if train == test:
                res = diagonal_ceiling(cells, preds, pipeline, recipe, train, bootstrap=bootstrap)
            else:
                res = cross_value(cells, preds, pipeline, recipe, train, test, bootstrap=bootstrap)
            row[test] = round(res["point"], 3) if res else None
        rows.append(row)
    return pd.DataFrame(rows)


def offdiagonal_mean(cells, preds, pipeline: str, recipe: str) -> float:
    """Mean subject-accuracy across the 6 off-diagonal transfers for a pipeline."""
    vals = []
    for train in SOURCES:
        for test in SOURCES:
            if train == test:
                continue
            res = cross_value(cells, preds, pipeline, recipe, train, test, bootstrap=False)
            if res:
                vals.append(res["point"])
    return float(np.mean(vals)) if vals else float("nan")
