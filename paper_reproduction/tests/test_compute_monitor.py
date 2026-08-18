"""Tests for ComputeMonitor's attributable memory and timing figures."""

import time

from parkinson_spiral.compute import ComputeMonitor, host_rss_peak_mb


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
    """
    import torch

    if torch.cuda.is_available():
        return
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
