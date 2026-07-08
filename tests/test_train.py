from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from parkinson_model_train.data import ImageRecord, TrainingConfig
from parkinson_model_train.reporting import write_metric_plots
from parkinson_model_train.train import DrawingDataset, _resolve_device, build_image_transforms


def test_build_image_transforms_uses_train_only_augmentation(tmp_path: Path) -> None:
    pytest.importorskip("torchvision")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_path,subject_id,label\n", encoding="utf-8")
    config = TrainingConfig(manifest_path=manifest, output_dir=tmp_path, image_size=32)

    train_transform, eval_transform = build_image_transforms(config)

    assert train_transform is not eval_transform
    assert "RandomResizedCrop" in str(train_transform)
    assert "RandomRotation" in str(train_transform)
    assert "GaussianBlur" in str(train_transform)
    assert "RandomResizedCrop" not in str(eval_transform)


def test_build_image_transforms_can_disable_augmentation(tmp_path: Path) -> None:
    pytest.importorskip("torchvision")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_path,subject_id,label\n", encoding="utf-8")
    config = TrainingConfig(manifest_path=manifest, output_dir=tmp_path, image_size=32, augment=False)

    train_transform, eval_transform = build_image_transforms(config)

    assert train_transform is eval_transform


def test_image_transform_returns_pytorch_tensor(tmp_path: Path) -> None:
    pytest.importorskip("torchvision")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_path,subject_id,label\n", encoding="utf-8")
    config = TrainingConfig(
        manifest_path=manifest,
        output_dir=tmp_path,
        image_size=32,
        rotation_degrees=0,
        blur_probability=0,
        jitter_probability=0,
    )
    train_transform, _ = build_image_transforms(config)
    image = Image.new("RGB", (48, 48), color="white")

    tensor = train_transform(image)

    assert tuple(tensor.shape) == (3, 32, 32)


def test_resolve_device_accepts_apple_mps() -> None:
    assert _resolve_device("mps").type == "mps"


def test_drawing_dataset_iterates_with_pytorch_dataloader(tmp_path: Path) -> None:
    image_path = tmp_path / "drawing.png"
    Image.new("RGB", (48, 48), color="white").save(image_path)

    def transform(_image: Image.Image) -> torch.Tensor:
        return torch.zeros((3, 32, 32))

    dataset = DrawingDataset([ImageRecord(image_path=image_path, subject_id="s1", label="PD")], {"PD": 0}, transform)

    images, labels = next(iter(DataLoader(dataset, batch_size=1)))

    assert tuple(images.shape) == (1, 3, 32, 32)
    assert labels.tolist() == [0]


def test_write_metric_plots_creates_standalone_svgs(tmp_path: Path) -> None:
    summary = {
        "split_counts": {"train": 2, "val": 2, "test": 2},
        "subject_counts": {"train": 2, "val": 2, "test": 2},
        "test_metrics": {"accuracy": 0.5, "balanced_accuracy": 0.5},
        "history": [{"epoch": 1, "train_loss": 0.7, "val_metrics": {"balanced_accuracy": 0.5}}],
    }

    paths = write_metric_plots(tmp_path, summary)

    assert {path.name for path in paths} == {"test_metrics.svg", "split_counts.svg", "validation_history.svg"}
    assert "balanced_accuracy" in (tmp_path / "test_metrics.svg").read_text(encoding="utf-8")
    assert "Split Counts" in (tmp_path / "split_counts.svg").read_text(encoding="utf-8")
    assert "Validation History" in (tmp_path / "validation_history.svg").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("rotation_degrees", -1, "rotation_degrees"),
        ("crop_scale_min", 0, "crop_scale_min"),
        ("crop_ratio_min", 2, "crop_ratio_min"),
        ("blur_probability", 2, "blur_probability"),
        ("blur_kernel_size", 2, "blur_kernel_size"),
        ("jitter_probability", -0.1, "jitter_probability"),
        ("brightness_jitter", -0.1, "brightness_jitter"),
        ("contrast_jitter", -0.1, "contrast_jitter"),
    ],
)
def test_training_config_validates_augmentation_options(
    tmp_path: Path, option: str, value: float | int, message: str
) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_path,subject_id,label\n", encoding="utf-8")
    config = TrainingConfig(manifest_path=manifest, output_dir=tmp_path, **{option: value})

    with pytest.raises(ValueError, match=message):
        config.validate()
