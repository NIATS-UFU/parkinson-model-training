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


def loader_kwargs() -> dict:
    """DataLoader worker settings that are safe on the host OS.

    Windows, macOS, and Python 3.14+ POSIX all use `spawn` for worker processes,
    which requires pickling the dataset and transforms. With spawn, persistent
    worker processes can deadlock against framework initialisation (e.g. TF CPU
    thread pools competing with PyTorch MPS on macOS). Linux `fork` avoids all of
    this, so we keep 2 workers only there.

    `SPIRAL_NUM_WORKERS` overrides the count. Set it to 0 on Linux for
    `dogan_triple`: forking a worker poisons OpenBLAS's thread pool, and Dogan's
    NCA then deadlocks in `scipy.linalg.lu` under `PCA._fit_truncated` once the
    feature matrix is large enough to cross OpenBLAS's multi-threading threshold
    (hemato, n_train=2699 x 5024). The smaller kaggle/folador Dogan cells stay
    single-threaded in BLAS and so survive. This path had never run before: the
    paper's Dogan cells ran on Windows and the evidence run on macOS, both of
    which take the `spawn` branch above and never fork.

    The worker count is not only a throughput knob -- the AUG-Huang and AUG-Dogan
    train transforms are stochastic and are drawn inside the worker, which seeds
    its RNG from the worker id -- so changing it changes results. Override per
    pipeline rather than globally."""
    import platform

    override = os.environ.get("SPIRAL_NUM_WORKERS")
    if override is not None:
        count = int(override)
        return {"num_workers": count} if count == 0 else {"num_workers": count, "persistent_workers": True}

    use_spawn = os.name == "nt" or platform.system() == "Darwin" or sys.version_info >= (3, 14)
    if use_spawn:
        return {"num_workers": 0}
    return {"num_workers": 2, "persistent_workers": True}


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
