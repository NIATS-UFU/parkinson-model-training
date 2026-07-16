"""Tests for the Huang VGG19 training safeguards."""

from parkinson_spiral.pipelines.huang_vgg19 import _uses_mixed_precision


def test_mixed_precision_requires_cuda_loss_scaling() -> None:
    """Use mixed precision only where the training loop applies loss scaling."""
    assert _uses_mixed_precision("cuda") is True
    assert _uses_mixed_precision("mps") is False
    assert _uses_mixed_precision("cpu") is False
