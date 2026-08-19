"""Tests for ComputeMonitor's attributable memory and timing figures."""

import time

import pytest
from parkinson_spiral.compute import ComputeMonitor, host_rss_peak_mb


def _cuda_available() -> bool:
    """Whether CUDA is usable here, tolerating torch being absent entirely.

    torch lives in the `[train]` extra, so the Layer-1 install these tests run
    under does not have it. `ComputeMonitor` handles that by catching
    ImportError, and so must anything asserting on its behaviour.
    """
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def test_wall_clock_is_measured() -> None:
    """The monitor times the block it wraps."""
    with ComputeMonitor() as monitor:
        time.sleep(0.01)
    assert monitor.wall_seconds is not None
    assert monitor.wall_seconds >= 0.01


def test_no_op_block_allocates_nothing() -> None:
    """A block that allocates no GPU memory reports no attributable growth.

    This is the defect the delta exists to fix: `gpu_peak_mb` alone reports
    whatever the process already holds, so a pipeline that never touches CUDA
    could still appear to use hundreds of megabytes.
    """
    with ComputeMonitor() as monitor:
        pass
    assert monitor.gpu_peak_delta_mb in (None, 0.0)


def test_gpu_fields_are_none_without_cuda() -> None:
    """Without CUDA the GPU figures are absent rather than zero.

    Zero would claim a measurement of no memory used; None says no measurement
    was possible. A cost table must be able to tell those apart.

    This covers both ways CUDA can be unavailable: no GPU on the host, and no
    torch installed at all. The second is the ordinary case for a Layer-1
    install, so it is the one that must not raise.
    """
    if _cuda_available():
        pytest.skip("CUDA is present; this asserts the no-measurement branch")
    with ComputeMonitor() as monitor:
        pass
    assert monitor.gpu_peak_mb is None
    assert monitor.gpu_peak_delta_mb is None
    assert monitor.gpu_peak_reserved_mb is None
    assert monitor.gpu_baseline_mb is None


def test_host_rss_is_positive_where_measurable() -> None:
    """`resource` is POSIX-only, so this is a real figure or an honest None."""
    peak = host_rss_peak_mb()
    assert peak is None or peak > 0
