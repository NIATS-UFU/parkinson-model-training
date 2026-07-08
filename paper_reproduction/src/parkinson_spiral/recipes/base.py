"""Recipe ABC: every preprocessing/augmentation pipeline conforms to this contract.

The harness instantiates a Recipe, wraps it with `PreprocessedSpiralDataset`, and
never has to know which paper the recipe came from. New recipes plug in by
subclassing `Recipe` and registering in `recipes/__init__.py`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from PIL import Image


@dataclass(frozen=True)
class RecipeMeta:
    name: str
    image_size: int
    normalization_mean: tuple[float, ...]
    normalization_std: tuple[float, ...]
    channels: int  # 3 for RGB, 1 for grayscale/skeleton


class Recipe(ABC):
    """Abstract preprocessing/augmentation pipeline.

    Three callable stages:
      - `pre_split_ops(img)`  : deterministic, applied to every image regardless of split.
      - `train_augment(img)`  : stochastic, applied only when `is_train=True`.
      - `eval_transform(img)` : deterministic, applied at eval (after pre_split_ops).

    The Dataset composes them as:
      img = pre_split_ops(img)
      img = train_augment(img) if is_train else eval_transform(img)
    """

    meta: RecipeMeta
    augment_then_split: bool = False

    @abstractmethod
    def pre_split_ops(self, img: Image.Image) -> Image.Image: ...

    @abstractmethod
    def train_augment(self, img: Image.Image) -> torch.Tensor: ...

    @abstractmethod
    def eval_transform(self, img: Image.Image) -> torch.Tensor: ...

    def variants_per_image(self) -> int:
        """How many deterministic copies a single source image produces when
        `augment_then_split=True`. Default 1; AUG-Dogan overrides to 8."""
        return 1

    def get_variant(self, img: Image.Image, variant_idx: int) -> torch.Tensor:
        """Produce the `variant_idx`-th deterministic copy (only meaningful in
        augment-then-split mode). Default ignores variant_idx."""
        return self.eval_transform(self.pre_split_ops(img))


def imagenet_normalize() -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def half_half_normalize() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Huang 2024 chose mean=0.5, std=0.5 per channel (non-standard for an
    ImageNet-pretrained VGG19, but kept for faithful reproduction)."""
    return (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
