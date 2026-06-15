"""AUG-None — control recipe (no augmentation).

Resize + ImageNet normalize only. Same in train and eval. This is the baseline
against which AUG-Huang / AUG-Dogan are compared.
"""
from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms

from .base import Recipe, RecipeMeta, imagenet_normalize


class AugNone(Recipe):
    def __init__(self, image_size: int = 224) -> None:
        mean, std = imagenet_normalize()
        self.meta = RecipeMeta(
            name="AUG-None",
            image_size=image_size,
            normalization_mean=mean,
            normalization_std=std,
            channels=3,
        )
        self._tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=list(mean), std=list(std)),
            ]
        )

    def pre_split_ops(self, img: Image.Image) -> Image.Image:
        return img.convert("RGB")

    def train_augment(self, img: Image.Image) -> torch.Tensor:
        return self._tf(img)

    def eval_transform(self, img: Image.Image) -> torch.Tensor:
        return self._tf(img)
