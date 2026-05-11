"""Training configuration for Parkinson drawing image models."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for frozen DINO image classifier training."""

    manifest_path: Path
    output_dir: Path
    image_size: int = 512
    model_name: str = "vit_small_patch14_dinov2"
    label_column: str = "label"
    image_column: str = "image_path"
    subject_column: str = "subject_id"
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    num_workers: int = 4
    device: str = "auto"

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
