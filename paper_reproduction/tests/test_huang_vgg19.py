"""Tests for the Huang VGG19 training safeguards."""

import numpy as np
import pytest
from parkinson_spiral.pipelines.huang_vgg19 import (
    _classify_gradient,
    _is_single_class_collapse,
    _uses_mixed_precision,
)


def test_mixed_precision_requires_cuda_loss_scaling() -> None:
    """Use mixed precision only where the training loop applies loss scaling."""
    assert _uses_mixed_precision("cuda") is True
    assert _uses_mixed_precision("mps") is False
    assert _uses_mixed_precision("cpu") is False


def test_usable_gradient_is_applied() -> None:
    """A finite, positive gradient norm yields a real optimizer update."""
    assert _classify_gradient(1.5, has_loss_scaler=True, epoch=0) == "usable"
    assert _classify_gradient(1.5, has_loss_scaler=False, epoch=0) == "usable"


def test_zero_gradient_is_skipped_not_fatal() -> None:
    """A vanished gradient is a benign no-op, not a collapse -- on either path.

    Near convergence a perfectly-fit batch has a genuinely ~0 gradient, and fp16
    can round a small-but-real gradient to exactly 0. Neither should crash.
    """
    assert _classify_gradient(0.0, has_loss_scaler=True, epoch=2) == "skip"
    assert _classify_gradient(0.0, has_loss_scaler=False, epoch=2) == "skip"


def test_nonfinite_recovered_under_loss_scaler() -> None:
    """Leave transient overflow to the dynamic loss scaler to back off and retry."""
    assert _classify_gradient(float("inf"), has_loss_scaler=True, epoch=2) == "skip"
    assert _classify_gradient(float("nan"), has_loss_scaler=True, epoch=2) == "skip"


def test_nonfinite_fatal_without_loss_scaler() -> None:
    """Reject non-finite gradients when no dynamic loss scaler is available."""
    with pytest.raises(FloatingPointError, match="non-finite gradients in epoch 3"):
        _classify_gradient(float("nan"), has_loss_scaler=False, epoch=2)


def test_single_class_collapse_detected() -> None:
    """A two-class training set predicted as one class is a real collapse."""
    predictions = np.zeros(10, dtype=int)
    assert _is_single_class_collapse(predictions, n_training_classes=2) is True


def test_two_class_predictions_are_not_collapse() -> None:
    """A model that predicts both classes on its training data has trained."""
    predictions = np.array([0, 1, 0, 1])
    assert _is_single_class_collapse(predictions, n_training_classes=2) is False


def test_single_class_training_data_is_not_collapse() -> None:
    """If the training labels were one class, single-class output is correct."""
    predictions = np.zeros(10, dtype=int)
    assert _is_single_class_collapse(predictions, n_training_classes=1) is False
