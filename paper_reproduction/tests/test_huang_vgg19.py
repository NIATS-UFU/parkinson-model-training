"""Tests for the Huang VGG19 training safeguards."""

import pytest
from parkinson_spiral.pipelines.huang_vgg19 import (
    _uses_mixed_precision,
    _validate_gradient_norm_value,
)


def test_mixed_precision_requires_cuda_loss_scaling() -> None:
    """Use mixed precision only where the training loop applies loss scaling."""
    assert _uses_mixed_precision("cuda") is True
    assert _uses_mixed_precision("mps") is False
    assert _uses_mixed_precision("cpu") is False


def test_zero_gradients_fail_fast() -> None:
    """Reject a vanished gradient before applying an optimizer update."""
    with pytest.raises(RuntimeError, match="zero gradients in epoch 3"):
        _validate_gradient_norm_value(0.0, recover_nonfinite=True, epoch=2)


def test_cuda_can_recover_from_nonfinite_gradients() -> None:
    """Leave transient CUDA overflow recovery to the dynamic loss scaler."""
    _validate_gradient_norm_value(float("inf"), recover_nonfinite=True, epoch=2)


def test_full_precision_rejects_nonfinite_gradients() -> None:
    """Reject non-finite gradients when no dynamic loss scaler is available."""
    with pytest.raises(FloatingPointError, match="non-finite gradients in epoch 3"):
        _validate_gradient_norm_value(float("nan"), recover_nonfinite=False, epoch=2)
