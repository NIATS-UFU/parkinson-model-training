"""Pipeline registry. Each module declares its class; this file collects them."""
from __future__ import annotations

import os
from typing import Type

from ..protocol import Pipeline


def loader_kwargs() -> dict:
    """DataLoader worker settings that are safe on the host OS.

    Windows (and Python 3.14+ POSIX) start workers with `spawn`, which pickles the
    dataset and recipe transforms — fragile and prone to worker-pool hangs — so we
    load single-process there. Linux uses `fork`, where workers are cheap and the
    GPU pipelines benefit, so we keep 2."""
    if os.name == "nt":
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


def get_pipeline_class(name: str) -> Type[Pipeline]:
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
