"""AUG-Huang — Huang et al. 2024 augmentation recipe.

Source: Huang et al. 2024 §3.2 (augmentation) + §9 (hyperparameters).

Verbatim spec + our documented defaults:
| Parameter        | Huang spec                         | Our default                    |
|------------------|------------------------------------|--------------------------------|
| Rotation degree  | "within a specified range"         | RandomRotation(+/-15 deg)      |
| HorizontalFlip   | yes (p inferred 0.5)               | RandomHorizontalFlip(p=0.5)    |
| VerticalFlip     | not listed                         | off                            |
| Image size       | 224x224                            | 224                            |
| Normalize        | mean=0.5/std=0.5 per channel       | mean=0.5/std=0.5 (faithful)    |
"""

import torch
from PIL import Image
from torchvision import transforms

from .base import Recipe, RecipeMeta, half_half_normalize, imagenet_normalize


class AugHuang(Recipe):
    def __init__(
        self,
        image_size: int = 224,
        rotation_degrees: float = 15.0,
        augmix: bool = False,
        use_imagenet_norm: bool = False,
    ) -> None:
        mean, std = imagenet_normalize() if use_imagenet_norm else half_half_normalize()
        self.meta = RecipeMeta(
            name="AUG-Huang",
            image_size=image_size,
            normalization_mean=mean,
            normalization_std=std,
            channels=3,
        )
        self.rotation_degrees = rotation_degrees
        self.augmix = augmix

        train_ops: list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=rotation_degrees),
        ]
        if augmix:
            train_ops.append(transforms.AugMix())
        train_ops += [
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
        self._train_tf = transforms.Compose(train_ops)

        self._eval_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ])

    def pre_split_ops(self, img: Image.Image) -> Image.Image:
        return img.convert("RGB")

    def train_augment(self, img: Image.Image) -> torch.Tensor:
        return self._train_tf(img)

    def eval_transform(self, img: Image.Image) -> torch.Tensor:
        return self._eval_tf(img)
