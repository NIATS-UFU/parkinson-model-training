"""DINOv2 (frozen) + LogisticRegression linear probe.

The release's headline model. Argument for H4: lowest free-hyperparameter surface
among all pipelines (3 params), no fine-tuning required, lightweight enough to run
CPU-only (the frozen backbone is a public download; only the linear probe is fit
by us). Falls back to `vit_size="vits14"` if vitb14 OOMs.
"""

import os
from pathlib import Path
from typing import ClassVar

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..protocol import Pipeline

_VIT_SIZES = {"vits14": 384, "vitb14": 768, "vitl14": 1024, "vitg14": 1536}
_DINOV2_HUB_REPO = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"


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
    _BACKBONE_CACHE: ClassVar[dict[tuple[str, str], object]] = {}
    _FEATURE_CACHE: ClassVar[dict[tuple[str, str], np.ndarray]] = {}

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

        from . import accelerator_device

        self._device = accelerator_device()
        cache_key = (self.vit_size, self._device)
        backbone = self._BACKBONE_CACHE.get(cache_key)
        if backbone is None:
            local_repo = os.environ.get("DINOV2_REPO_DIR")
            local_weights = os.environ.get("DINOV2_WEIGHTS")
            if bool(local_repo) != bool(local_weights):
                raise ValueError("DINOV2_REPO_DIR and DINOV2_WEIGHTS must be set together")
            if local_repo and local_weights:
                repo_path = Path(local_repo)
                weights_path = Path(local_weights)
                if not repo_path.is_dir():
                    raise FileNotFoundError(f"DINOv2 source directory not found: {repo_path}")
                if not weights_path.is_file():
                    raise FileNotFoundError(f"DINOv2 weights not found: {weights_path}")
                backbone = torch.hub.load(
                    str(repo_path),
                    f"dinov2_{self.vit_size}",
                    source="local",
                    weights=str(weights_path),
                )
            else:
                backbone = torch.hub.load(_DINOV2_HUB_REPO, f"dinov2_{self.vit_size}")
            backbone.eval()
            for p in backbone.parameters():
                p.requires_grad = False
            backbone = backbone.to(self._device)
            self._BACKBONE_CACHE[cache_key] = backbone
        self._backbone = backbone

    @staticmethod
    def _can_cache_features(dataset) -> bool:
        """Return whether the dataset transform is deterministic per image."""
        recipe_name = getattr(getattr(dataset, "recipe", None), "meta", None)
        return getattr(recipe_name, "name", None) == "AUG-None" and not getattr(dataset, "augment_then_split", False)

    def _extract(self, dataset) -> tuple[np.ndarray, np.ndarray]:
        import torch
        from torch.utils.data import DataLoader

        self._ensure_backbone()
        from . import loader_kwargs

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=(self._device == "cuda"),
            **loader_kwargs(),
        )
        feats: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        use_cache = self._can_cache_features(dataset)
        with torch.inference_mode():
            for tensors, label_ints, image_ids in loader:
                batch_features: list[np.ndarray | None] = [None] * len(image_ids)
                missing_indices: list[int] = []
                if use_cache:
                    for index, image_id in enumerate(image_ids):
                        cached = self._FEATURE_CACHE.get((self.vit_size, str(image_id)))
                        if cached is None:
                            missing_indices.append(index)
                        else:
                            batch_features[index] = cached
                else:
                    missing_indices = list(range(len(image_ids)))

                if missing_indices:
                    missing_tensors = tensors[missing_indices].to(self._device, non_blocking=True)
                    missing_features = self._backbone(missing_tensors).detach().cpu().numpy()
                    for index, feature in zip(missing_indices, missing_features, strict=True):
                        batch_features[index] = feature
                        if use_cache:
                            self._FEATURE_CACHE[(self.vit_size, str(image_ids[index]))] = feature

                feats.append(np.stack(batch_features, axis=0))
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
