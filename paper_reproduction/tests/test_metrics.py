"""Binary metrics, subject aggregation, bootstrap CI."""

import numpy as np
import pandas as pd
import pytest
from parkinson_spiral.metrics import (
    aggregate_to_subject,
    binary_metrics,
    percentile_bootstrap_ci,
)


def test_binary_metrics_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.9, 0.8])
    m = binary_metrics(y, p)
    assert m["accuracy"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["auc"] == 1.0


def test_binary_metrics_pd_is_positive_class():
    # Only the PD (label 1) cases are missed -> sensitivity 0, specificity 1.
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.1, 0.2])
    m = binary_metrics(y, p)
    assert m["sensitivity"] == 0.0
    assert m["specificity"] == 1.0


def test_auc_nan_for_single_class():
    m = binary_metrics(np.array([1, 1, 1]), np.array([0.2, 0.7, 0.9]))
    assert np.isnan(m["auc"])


def test_aggregate_to_subject_means_proba_and_keeps_label():
    df = pd.DataFrame({
        "subject_id": ["s1", "s1", "s2"],
        "y_true": [1, 1, 0],
        "y_proba_pos": [0.4, 0.8, 0.2],
    })
    subj = aggregate_to_subject(df)
    assert len(subj) == 2
    s1 = subj[subj["subject_id"] == "s1"].iloc[0]
    assert s1["y_proba_pos"] == pytest.approx(0.6)
    assert s1["y_true"] == 1
    assert s1["n_images"] == 2


def test_bootstrap_ci_is_deterministic_and_brackets_point():
    df = pd.DataFrame({
        "y_true": [0, 0, 1, 1, 0, 1, 1, 0],
        "y_proba_pos": [0.1, 0.3, 0.9, 0.6, 0.2, 0.7, 0.8, 0.4],
    })

    def acc(yt, yp):
        return float((yt == (yp >= 0.5).astype(int)).mean())

    a = percentile_bootstrap_ci(df, acc, seed=20260516)
    b = percentile_bootstrap_ci(df, acc, seed=20260516)
    assert a == b  # same seed -> identical
    assert a["ci_lo"] <= a["point"] <= a["ci_hi"]
