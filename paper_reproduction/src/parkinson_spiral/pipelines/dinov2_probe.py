"""DINOv2 (frozen) + LogisticRegression linear probe.

The release's headline model. Argument for H4: lowest free-hyperparameter surface
among all pipelines (3 params), no fine-tuning required, lightweight enough to run
CPU-only (the frozen backbone is a public download; only the linear probe is fit
by us). Falls back to `vit_size="vits14"` if vitb14 OOMs.
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..protocol import Pipeline


_VIT_SIZES = {"vits14": 384, "vitb14": 768, "vitl14": 1024, "vitg14": 1536}


class DinoV2Probe(Pipeline):
    name: ClassVar[str] = "dinov2_probe"
    FREE_HYPERPARAMETERS: ClassVar[list[str]] = ["vit_size", "probe_C", "image_size"]
    METHODS_SECTION_NOTES: ClassVar[str] = (
        "DINOv2 (Oquab et al., 2023) is used as a frozen feature extractor. We use the "
        "vitb14 variant (86M parameters, 768-d CLS token) downloaded from the "
        "facebookresearch/dinov2 hub. No fine-tuning or LoRA is applied. The probe is a "
        "scikit-learn LogisticRegression with default C=1.0 and max_iter=2000. The choice "
        "of frozen + linear probe is deliberate: it minimises the free-hyperparameter "
        "surface against which a clinical practitioner would need to tune, which is the "
        "core of our H4 reproducibility-cost argument. If vitb14 exceeds available GPU "
        "memory, the harness falls back to vits14 (22M params, 384-d CLS token) and the "
        "size-vs-accuracy trade-off is reported."
    )

    def __init__(
        self,
        vit_size: str = "vitb14",
        probe_C: float = 1.0,
        batch_size: int = 32,
    ) -> None:
        if vit_size not in _VIT_SIZES:
            raise ValueError(f"vit_size must be one of {list(_VIT_SIZES)}; got {vit_size!r}")
        self.vit_size = vit_size
        self.probe_C = probe_C
        self.batch_size = batch_size
        self._backbone = None
        self._probe: LogisticRegression | None = None

    def _ensure_backbone(self):
        if self._backbone is not None:
            return
        import torch
        backbone = torch.hub.load("facebookresearch/dinov2", f"dinov2_{self.vit_size}")
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._backbone = backbone.to(self._device)

    def _extract(self, dataset) -> tuple[np.ndarray, np.ndarray]:
        import torch
        from torch.utils.data import DataLoader

        self._ensure_backbone()
        from . import loader_kwargs
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            pin_memory=(self._device == "cuda"), **loader_kwargs(),
        )
        feats: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        with torch.inference_mode():
            for tensors, label_ints, _image_ids in loader:
                tensors = tensors.to(self._device, non_blocking=True)
                cls = self._backbone(tensors)  # (B, D)
                feats.append(cls.detach().cpu().numpy())
                labels.append(np.asarray(label_ints))
        return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)

    def fit(self, dataset, *, seed: int) -> None:
        X, y = self._extract(dataset)
        self._probe = LogisticRegression(C=self.probe_C, max_iter=2000, random_state=seed)
        self._probe.fit(X, y)

    def predict_proba(self, dataset) -> np.ndarray:
        if self._probe is None:
            raise RuntimeError("predict_proba called before fit")
        X, _ = self._extract(dataset)
        return self._probe.predict_proba(X)
