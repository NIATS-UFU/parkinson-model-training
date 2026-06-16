"""Classification metrics + percentile bootstrap CI + multi-level aggregation.

PD = positive class. All reported metrics are subject-level for every cell;
image/session aggregations exist as internal building blocks only. Kaggle
subject IDs are filename-derived (mostly 1 image/subject), so its subject-level
aggregation largely coincides with per-image scoring.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import DEFAULT_SEED


def binary_metrics(y_true: np.ndarray, y_proba_pos: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """PD is positive class. Returns {accuracy, sensitivity, specificity, precision, f1, auc}."""
    y_pred = (y_proba_pos >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    # AUC is undefined for single-class test sets.
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, y_proba_pos))
    else:
        out["auc"] = float("nan")
    return out


def percentile_bootstrap_ci(
    df_pred: pd.DataFrame,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    """Percentile bootstrap resampling the rows of df_pred (must have y_true, y_proba_pos).
    Reported callers pass a subject-aggregated frame, so resampling is over subjects."""
    rng = np.random.default_rng(seed)
    n = len(df_pred)
    y_true = df_pred["y_true"].to_numpy()
    y_proba = df_pred["y_proba_pos"].to_numpy()
    point = metric_fn(y_true, y_proba)
    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = metric_fn(y_true[idx], y_proba[idx])
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return {"point": float(point), "ci_lo": lo, "ci_hi": hi}


def aggregate_to_session(df_pred: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Mean P(PD) + majority-vote per (subject_id, session_id). Drops rows with null session_id."""
    s = df_pred.dropna(subset=["session_id"])
    if s.empty:
        return s.assign(y_pred=pd.Series(dtype=int))
    grouped = (
        s.groupby(["subject_id", "session_id"])
        .agg(
            y_proba_pos=("y_proba_pos", "mean"),
            y_true=("y_true", "first"),
            n_images=("y_true", "size"),
        )
        .reset_index()
    )
    grouped["y_pred"] = (grouped["y_proba_pos"] >= threshold).astype(int)
    return grouped


def aggregate_to_subject(df_pred: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Mean P(PD) + majority-vote per subject_id."""
    grouped = (
        df_pred.groupby("subject_id")
        .agg(
            y_proba_pos=("y_proba_pos", "mean"),
            y_true=("y_true", "first"),
            n_images=("y_true", "size"),
        )
        .reset_index()
    )
    grouped["y_pred"] = (grouped["y_proba_pos"] >= threshold).astype(int)
    return grouped


def metrics_at_all_levels(df_pred: pd.DataFrame) -> dict[str, dict[str, float] | None]:
    """Image + session + subject aggregations of binary_metrics."""
    out: dict[str, dict[str, float] | None] = {
        "image": binary_metrics(df_pred["y_true"].to_numpy(), df_pred["y_proba_pos"].to_numpy())
    }
    sess = aggregate_to_session(df_pred)
    out["session"] = (
        binary_metrics(sess["y_true"].to_numpy(), sess["y_proba_pos"].to_numpy()) if not sess.empty else None
    )
    subj = aggregate_to_subject(df_pred)
    out["subject"] = binary_metrics(subj["y_true"].to_numpy(), subj["y_proba_pos"].to_numpy())
    return out
