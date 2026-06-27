import pytest

from parkinson_model_train.evaluation import compute_metrics, compute_roc_auc


def test_compute_metrics_requires_samples() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        compute_metrics([], [], [])


def test_compute_roc_auc_requires_samples() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        compute_roc_auc([], [])
