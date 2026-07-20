"""Pipeline registry. Each module declares its class; this file collects them."""

import os
import sys

from ..protocol import Pipeline


def accelerator_device() -> str:
    """Return the best available PyTorch accelerator, including Apple MPS."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_loader_kwargs(
    *,
    os_name: str,
    system: str,
    python_version: tuple[int, int],
    forbid_fork: bool,
    override: str | None,
) -> dict:
    """Pure decision behind `loader_kwargs`, separated from the platform/env reads
    so every branch is testable without touching the real host.

    Precedence:
    1. `SPIRAL_NUM_WORKERS`, if set, wins unconditionally -- an explicit escape
       hatch (force a worker count, or reproduce the deadlock below for debugging).
    2. Windows, macOS, and Python 3.14+ use `spawn`, which is expensive to start and
       can deadlock against framework init (TF CPU pools vs PyTorch MPS on macOS), so
       we run single-process there -- also matching the published Windows/macOS runs.
    3. On Linux we fork 2 workers, EXCEPT when a pipeline passes `forbid_fork=True`.
       Forking poisons OpenBLAS's threadpool; a subsequent large BLAS call
       (dogan_triple's NCA -> `scipy.linalg.lu` under `PCA._fit_truncated`) then
       deadlocks silently on multi-core hosts. 0 workers avoids the fork and matches
       the worker count Dogan's published numbers were produced with.
    """
    if override is not None:
        count = int(override)
        if count == 0:
            return {"num_workers": 0}
        return {"num_workers": count, "persistent_workers": True}

    use_spawn = os_name == "nt" or system == "Darwin" or python_version >= (3, 14)
    if use_spawn or forbid_fork:
        return {"num_workers": 0}
    return {"num_workers": 2, "persistent_workers": True}


def loader_kwargs(*, forbid_fork: bool = False) -> dict:
    """DataLoader worker settings that are safe on the host OS.

    Pass `forbid_fork=True` from a pipeline that makes large BLAS calls once the
    DataLoader has started (see `_resolve_loader_kwargs` for the OpenBLAS fork
    deadlock); it forces single-process loading on Linux instead of forking. The
    `SPIRAL_NUM_WORKERS` environment variable overrides everything. Defaults are
    unchanged for every pipeline that does not opt in."""
    import platform

    return _resolve_loader_kwargs(
        os_name=os.name,
        system=platform.system(),
        python_version=sys.version_info[:2],
        forbid_fork=forbid_fork,
        override=os.environ.get("SPIRAL_NUM_WORKERS"),
    )


# Lazy-loaded: importing torch/timm/tensorflow at the registry level would require
# all of them installed even for trivial commands, so we resolve by name on demand.
_PIPELINE_NAMES = ("dinov2_probe", "folador_hog", "huang_vgg19", "dogan_triple")


def get_pipeline(name: str, **kwargs) -> Pipeline:
    if name not in _PIPELINE_NAMES:
        raise ValueError(f"unknown pipeline {name!r}; available: {_PIPELINE_NAMES}")
    if name == "dinov2_probe":
        from .dinov2_probe import DinoV2Probe

        return DinoV2Probe(**kwargs)
    if name == "folador_hog":
        from .folador_hog import FoladorHOG

        return FoladorHOG(**kwargs)
    if name == "huang_vgg19":
        from .huang_vgg19 import HuangVGG19

        return HuangVGG19(**kwargs)
    if name == "dogan_triple":
        from .dogan_triple import DoganTriple

        return DoganTriple(**kwargs)
    raise AssertionError("unreachable")


def get_pipeline_class(name: str) -> type[Pipeline]:
    """Return the class (without instantiating) — used for FREE_HYPERPARAMETERS / METHODS_SECTION_NOTES."""
    if name == "dinov2_probe":
        from .dinov2_probe import DinoV2Probe

        return DinoV2Probe
    if name == "folador_hog":
        from .folador_hog import FoladorHOG

        return FoladorHOG
    if name == "huang_vgg19":
        from .huang_vgg19 import HuangVGG19

        return HuangVGG19
    if name == "dogan_triple":
        from .dogan_triple import DoganTriple

        return DoganTriple
    raise ValueError(f"unknown pipeline {name!r}")


def list_pipeline_names() -> list[str]:
    return list(_PIPELINE_NAMES)
