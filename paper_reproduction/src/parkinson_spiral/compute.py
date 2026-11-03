"""ComputeMonitor: wall-clock, attributable CUDA peak memory, and host RSS.

Used as a context manager around `pipeline.fit` and `pipeline.predict_proba`
inside `runner.run_cell`. Falls back to None values when CUDA is unavailable.

Three memory figures are recorded because they answer different questions, and
conflating them is how a benchmark process misreports a pipeline's footprint:

- `gpu_peak_mb` is the process-wide CUDA high-water mark. A benchmark process
  runs four pipelines and keeps backbones cached between cells, so this figure
  includes memory the measured block never asked for. It is kept for continuity
  with earlier runs; it is not attributable to one pipeline.
- `gpu_peak_delta_mb` subtracts the allocation standing at entry, so it is the
  growth this block actually caused. This is the attributable number.
- `gpu_peak_reserved_mb` is what the caching allocator took from the card, which
  is the figure that has to fit on the GPU.

`host_rss_peak_mb` is a process-lifetime high-water mark rather than a per-block
measurement -- `ru_maxrss` only ever rises -- so it is meaningful only in a
process running a single pipeline.
"""

import sys
import time

_BYTES_PER_MB = 2**20
_KIB_PER_MB = 1024


def host_rss_peak_mb() -> float | None:
    """Peak resident set size of this process in MiB, or None where unmeasurable.

    `resource` is POSIX-only, so this returns None on Windows. `ru_maxrss` is
    reported in kibibytes on Linux and in bytes on macOS.
    """
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / _BYTES_PER_MB if sys.platform == "darwin" else peak / _KIB_PER_MB


class ComputeMonitor:
    def __init__(self) -> None:
        self.wall_seconds: float | None = None
        self.gpu_baseline_mb: float | None = None
        self.gpu_peak_mb: float | None = None
        self.gpu_peak_delta_mb: float | None = None
        self.gpu_peak_reserved_mb: float | None = None
        self.host_rss_peak_mb: float | None = None

    def __enter__(self) -> "ComputeMonitor":
        try:
            import torch

            self._cuda = torch.cuda.is_available()
        except ImportError:
            self._cuda = False
        self._baseline_bytes = 0
        if self._cuda:
            import torch

            # Read the standing allocation before resetting, so the block's own
            # growth can be separated from what the process already held.
            self._baseline_bytes = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.wall_seconds = time.perf_counter() - self._t0
        if self._cuda:
            import torch

            peak_bytes = torch.cuda.max_memory_allocated()
            self.gpu_baseline_mb = self._baseline_bytes / _BYTES_PER_MB
            self.gpu_peak_mb = peak_bytes / _BYTES_PER_MB
            self.gpu_peak_delta_mb = max(peak_bytes - self._baseline_bytes, 0) / _BYTES_PER_MB
            self.gpu_peak_reserved_mb = torch.cuda.max_memory_reserved() / _BYTES_PER_MB
        self.host_rss_peak_mb = host_rss_peak_mb()
