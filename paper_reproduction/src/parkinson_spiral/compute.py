"""ComputeMonitor: wall-clock + CUDA peak memory.

Used as a context manager around `pipeline.fit` and `pipeline.predict_proba`
inside `runner.run_cell`. Falls back to None values when CUDA is unavailable.
"""

import time


class ComputeMonitor:
    def __init__(self) -> None:
        self.wall_seconds: float | None = None
        self.gpu_peak_mb: float | None = None

    def __enter__(self) -> "ComputeMonitor":
        try:
            import torch

            self._cuda = torch.cuda.is_available()
            if self._cuda:
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            self._cuda = False
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.wall_seconds = time.perf_counter() - self._t0
        if self._cuda:
            import torch

            self.gpu_peak_mb = torch.cuda.max_memory_allocated() / (2**20)
