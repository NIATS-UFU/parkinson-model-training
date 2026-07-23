"""Tests for the Huang VGG19 training safeguards."""

import numpy as np
import pytest
from parkinson_spiral.pipelines.huang_vgg19 import (
    _classify_gradient,
    _is_single_class_collapse,
)


def test_usable_gradient_is_applied() -> None:
    """A finite, positive gradient norm yields a real optimizer update."""
    assert _classify_gradient(1.5, epoch=0) == "usable"


def test_zero_gradient_is_skipped_not_fatal() -> None:
    """A vanished gradient is a benign no-op, not a collapse.

    Near convergence a perfectly-fit batch has a genuinely ~0 gradient; stepping
    on it would be a wasted update, not a crash.
    """
    assert _classify_gradient(0.0, epoch=2) == "skip"


def test_nonfinite_gradient_is_fatal() -> None:
    """bfloat16 carries fp32's range, so an inf/NaN gradient is a real breakdown."""
    with pytest.raises(FloatingPointError, match="non-finite gradients in epoch 3"):
        _classify_gradient(float("nan"), epoch=2)
    with pytest.raises(FloatingPointError, match="non-finite gradients in epoch 3"):
        _classify_gradient(float("inf"), epoch=2)


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
