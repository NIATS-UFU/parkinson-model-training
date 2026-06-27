"""Dataclasses used by the training data pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for frozen DINO image classifier training."""

    manifest_path: Path
    output_dir: Path
    image_size: int = 518
    model_name: str = "vit_small_patch14_dinov2"
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    num_workers: int = 4
    device: str = "auto"
    augment: bool = True
    rotation_degrees: float = 10.0
    crop_scale_min: float = 0.9
    crop_ratio_min: float = 0.95
    crop_ratio_max: float = 1.05
    blur_probability: float = 0.1
    blur_kernel_size: int = 3
    jitter_probability: float = 0.2
    brightness_jitter: float = 0.08
    contrast_jitter: float = 0.08

    def validate(self) -> None:
        """Raise ValueError if the training configuration is invalid."""
        if not self.manifest_path.exists():
            msg = f"Manifest does not exist: {self.manifest_path}"
            raise ValueError(msg)
        if self.image_size <= 0:
            msg = "image_size must be positive"
            raise ValueError(msg)
        if self.batch_size <= 0:
            msg = "batch_size must be positive"
            raise ValueError(msg)
        if self.epochs <= 0:
            msg = "epochs must be positive"
            raise ValueError(msg)
        if not 0 < self.learning_rate:
            msg = "learning_rate must be positive"
            raise ValueError(msg)
        if not 0 <= self.weight_decay:
            msg = "weight_decay cannot be negative"
            raise ValueError(msg)
        if not 0 <= self.val_fraction < 1:
            msg = "val_fraction must be in [0, 1)"
            raise ValueError(msg)
        if not 0 <= self.test_fraction < 1:
            msg = "test_fraction must be in [0, 1)"
            raise ValueError(msg)
        if self.val_fraction + self.test_fraction >= 1:
            msg = "val_fraction + test_fraction must be less than 1"
            raise ValueError(msg)
        if self.num_workers < 0:
            msg = "num_workers cannot be negative"
            raise ValueError(msg)
        if self.rotation_degrees < 0:
            msg = "rotation_degrees cannot be negative"
            raise ValueError(msg)
        if not 0 < self.crop_scale_min <= 1:
            msg = "crop_scale_min must be in (0, 1]"
            raise ValueError(msg)
        if not 0 < self.crop_ratio_min <= self.crop_ratio_max:
            msg = "crop_ratio_min and crop_ratio_max must be positive and ordered"
            raise ValueError(msg)
        if not 0 <= self.blur_probability <= 1:
            msg = "blur_probability must be in [0, 1]"
            raise ValueError(msg)
        if self.blur_kernel_size <= 0 or self.blur_kernel_size % 2 == 0:
            msg = "blur_kernel_size must be a positive odd integer"
            raise ValueError(msg)
        if not 0 <= self.jitter_probability <= 1:
            msg = "jitter_probability must be in [0, 1]"
            raise ValueError(msg)
        if self.brightness_jitter < 0:
            msg = "brightness_jitter cannot be negative"
            raise ValueError(msg)
        if self.contrast_jitter < 0:
            msg = "contrast_jitter cannot be negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class ImageRecord:
    """One labelled image and its subject identifier."""

    image_path: Path
    subject_id: str
    label: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSplit:
    """Train, validation, and test records split by subject."""

    train: list[ImageRecord]
    val: list[ImageRecord]
    test: list[ImageRecord]
