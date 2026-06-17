"""Core contracts: CellSpec, Pipeline ABC.

Every model the harness can run is a `Pipeline` subclass exposing
`fit(dataset)` + `predict_proba(dataset)`. The harness is framework-agnostic.

A `CellSpec` fully identifies one experiment. Its `cell_id` is a SHA1 over the
canonical JSON of the spec, so a cell's identity is derived purely from its
parameters — the same spec always maps to the same id, and generated result
files in `results/` are keyed by it. The hash formula is frozen: changing it
would break the link between the catalog and generated results.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

import numpy as np

# The three datasets, named as in the paper: the public Kaggle and Folador
# (Folador et al. 2021) spiral datasets and the Hemato clinical cohort.
VALID_SOURCES = {"kaggle", "folador", "hemato"}

# The four split protocols the paper uses.
VALID_SPLIT_SOURCES = {"published", "subject-clean", "cv-fold", "cross-dataset"}


@dataclass(frozen=True)
class CellSpec:
    pipeline_name: str
    recipe_name: str  # 'AUG-None' | 'AUG-Huang' | 'AUG-Dogan'
    train_source: str  # 'kaggle' | 'folador' | 'hemato'
    test_source: str
    split_source: str  # see VALID_SPLIT_SOURCES
    cv_fold: int | None  # required iff split_source == 'cv-fold'
    task_filter: str  # 'spiral' (the paper uses spiral drawings only)
    target_column: str  # 'class_label' (binary PD vs healthy)
    augment_then_split: bool  # literature's leaky augment-before-split order (one fidelity cell)
    seed: int
    pipeline_kwargs: dict[str, Any] = field(default_factory=dict)
    recipe_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def cell_id(self) -> str:
        payload = asdict(self)
        canonical = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]

    def validate(self) -> None:
        if self.split_source not in VALID_SPLIT_SOURCES:
            raise ValueError(f"unknown split_source {self.split_source!r}; valid: {sorted(VALID_SPLIT_SOURCES)}")
        if self.target_column != "class_label":
            raise ValueError(f"target_column must be 'class_label' (binary PD task); got {self.target_column!r}")
        if self.split_source == "cv-fold":
            if self.cv_fold is None:
                raise ValueError("cv_fold required when split_source=='cv-fold'")
            if self.train_source != self.test_source:
                raise ValueError("cv-fold split-source is within-dataset (train_source == test_source)")
            if self.train_source not in VALID_SOURCES:
                raise ValueError(f"cv-fold valid only for {sorted(VALID_SOURCES)} diagonals")
        if self.split_source == "cross-dataset":
            # Off-diagonal transfer only: train on ALL of A, test on ALL of B. A within-dataset
            # cell would test on training images, so it must use a real split instead.
            if self.train_source == self.test_source:
                raise ValueError("cross-dataset split requires train_source != test_source (off-diagonal only)")
            if self.cv_fold is not None:
                raise ValueError("cross-dataset split takes no cv_fold")
        if self.task_filter not in {"spiral", "wave", "both"}:
            raise ValueError(f"unknown task_filter {self.task_filter!r}")


class Pipeline(ABC):
    """Framework-agnostic training contract.

    Subclasses declare `FREE_HYPERPARAMETERS` and `METHODS_SECTION_NOTES` as
    class variables. The length of `FREE_HYPERPARAMETERS` is what the paper's H4
    "reproducibility cost" argument counts (DINOv2=3, Folador HOG=4,
    Dogan triple=11, Huang VGG19=12).
    """

    name: ClassVar[str]
    FREE_HYPERPARAMETERS: ClassVar[list[str]]
    METHODS_SECTION_NOTES: ClassVar[str]

    @abstractmethod
    def fit(self, dataset, *, seed: int) -> None:
        """Train the model on `dataset` (a PreprocessedSpiralDataset)."""

    @abstractmethod
    def predict_proba(self, dataset) -> np.ndarray:
        """Return shape (N, 2) array; column 1 is P(positive class = parkinson)."""
