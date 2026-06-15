"""AUG-Dogan — Dogan et al. 2024 augmentation recipe.

Source: Dogan et al. 2024 §2.

Verbatim spec + our documented defaults:
| Parameter       | Dogan spec                              | Our default                          |
|-----------------|-----------------------------------------|--------------------------------------|
| Rotation        | {90,180,270} (fixed, deterministic)     | same (emitted as deterministic copies)|
| Horizontal flip | yes                                     | yes                                  |
| Vertical flip   | yes                                     | yes                                  |
| Color transform | "color transformations" (unspecified)   | ColorJitter(0.2, 0.2, 0.2)           |
| Image size      | 224x224                                 | 224                                  |
| Order           | "augmentation ... then split"           | controlled by augment_then_split     |
| Norm            | not stated                              | ImageNet mean/std                    |

In `augment_then_split=True` mode this recipe produces 8 deterministic copies per
source image (used only by the one Dogan fidelity cell that reproduces the
literature's augment-before-split order).
"""
from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from .base import Recipe, RecipeMeta, imagenet_normalize


class _RotateBy:
    """Picklable fixed-angle rotation. A bare lambda is a local object that cannot be
    pickled, which breaks DataLoader(num_workers>0) on spawn platforms (Windows,
    Python 3.14+ POSIX). A module-level callable pickles fine."""

    def __init__(self, angle: int) -> None:
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle) if self.angle else x


# 8 deterministic (rotation, hflip, vflip) variants.
_VARIANTS: list[tuple[int, bool, bool]] = [
    (0, False, False),
    (90, False, False),
    (180, False, False),
    (270, False, False),
    (0, True, False),
    (90, True, False),
    (0, False, True),
    (90, False, True),
]


class AugDogan(Recipe):
    def __init__(
        self,
        image_size: int = 224,
        color_brightness: float = 0.2,
        color_contrast: float = 0.2,
        color_saturation: float = 0.2,
        augment_then_split: bool = False,
    ) -> None:
        mean, std = imagenet_normalize()
        self.meta = RecipeMeta(
            name="AUG-Dogan",
            image_size=image_size,
            normalization_mean=mean,
            normalization_std=std,
            channels=3,
        )
        self.augment_then_split = augment_then_split

        self._color = transforms.ColorJitter(
            brightness=color_brightness, contrast=color_contrast, saturation=color_saturation
        )

        train_ops = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomChoice(
                [_RotateBy(0), _RotateBy(90), _RotateBy(180), _RotateBy(270)]
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            self._color,
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
        self._train_tf = transforms.Compose(train_ops)

        self._eval_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=list(mean), std=list(std)),
            ]
        )
        # Deterministic-variant mode: rotation/flip already applied by caller,
        # so only the normalize stage is needed.
        self._variant_tail = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=list(mean), std=list(std)),
            ]
        )

    def pre_split_ops(self, img: Image.Image) -> Image.Image:
        return img.convert("RGB")

    def train_augment(self, img: Image.Image) -> torch.Tensor:
        return self._train_tf(img)

    def eval_transform(self, img: Image.Image) -> torch.Tensor:
        return self._eval_tf(img)

    def variants_per_image(self) -> int:
        return len(_VARIANTS) if self.augment_then_split else 1

    def get_variant(self, img: Image.Image, variant_idx: int) -> torch.Tensor:
        rgb = img.convert("RGB")
        rgb = TF.resize(rgb, [self.meta.image_size, self.meta.image_size])
        angle, hflip, vflip = _VARIANTS[variant_idx]
        if angle:
            rgb = TF.rotate(rgb, angle)
        if hflip:
            rgb = TF.hflip(rgb)
        if vflip:
            rgb = TF.vflip(rgb)
        return self._variant_tail(rgb)
