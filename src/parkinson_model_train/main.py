"""Command line interface for Parkinson drawing model training."""

from pathlib import Path
from typing import Annotated

import typer

from parkinson_model_train.config import TrainingConfig
from parkinson_model_train.data import (
    labels_to_indices,
    load_dataset_records,
    load_manifest,
    split_by_subject,
    validate_image_paths,
    write_manifest,
)
from parkinson_model_train.train import train_classifier

app = typer.Typer(help="Train frozen DINO classifiers for Parkinson drawing crops.")


@app.command()
def validate_manifest(
    manifest: Annotated[Path, typer.Argument(help="CSV with image_path, subject_id, and label columns.")],
    check_files: Annotated[bool, typer.Option(help="Verify that all image paths exist.")] = True,
) -> None:
    """Validate the input manifest and print its class/subject summary."""
    records = load_manifest(manifest)
    if check_files:
        validate_image_paths(records)
    labels = labels_to_indices(records)
    typer.echo(
        f"records={len(records)} subjects={len({record.subject_id for record in records})} labels={list(labels)}"
    )


@app.command()
def split_summary(
    manifest: Annotated[Path, typer.Argument(help="CSV with image_path, subject_id, and label columns.")],
    val_fraction: Annotated[float, typer.Option(help="Subject fraction for validation.")] = 0.15,
    test_fraction: Annotated[float, typer.Option(help="Subject fraction for testing.")] = 0.15,
    seed: Annotated[int, typer.Option(help="Random seed for subject split.")] = 42,
) -> None:
    """Print leakage-safe train/validation/test split sizes."""
    records = load_manifest(manifest)
    split = split_by_subject(records, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed)
    typer.echo(f"train={len(split.train)} val={len(split.val)} test={len(split.test)}")


@app.command()
def create_manifest(
    dataset: Annotated[
        str,
        typer.Argument(
            help=(
                "Dataset loader: hemato, handpd, newhandpd, kaggle-drawings, kaggle-spiral, "
                "roboflow, folador, or image-tree."
            )
        ),
    ],
    root: Annotated[Path, typer.Argument(help="Dataset root directory.")],
    output: Annotated[Path, typer.Argument(help="Output CSV manifest path.")],
) -> None:
    """Create a normalized image manifest from a supported dataset layout."""
    records = load_dataset_records(dataset, root)
    write_manifest(records, output)
    labels = labels_to_indices(records)
    typer.echo(f"wrote manifest: {output} records={len(records)} labels={list(labels)}")


@app.command()
def train(
    manifest: Annotated[Path, typer.Argument(help="CSV with image_path, subject_id, and label columns.")],
    output_dir: Annotated[Path, typer.Argument(help="Directory for checkpoints and run summaries.")],
    epochs: Annotated[int, typer.Option(help="Number of classifier training epochs.")] = 10,
    batch_size: Annotated[int, typer.Option(help="Training batch size.")] = 16,
    learning_rate: Annotated[float, typer.Option(help="AdamW learning rate.")] = 1e-3,
    weight_decay: Annotated[float, typer.Option(help="AdamW weight decay.")] = 1e-4,
    val_fraction: Annotated[float, typer.Option(help="Subject fraction for validation.")] = 0.15,
    test_fraction: Annotated[float, typer.Option(help="Subject fraction for testing.")] = 0.15,
    seed: Annotated[int, typer.Option(help="Random seed for subject split.")] = 42,
    num_workers: Annotated[int, typer.Option(help="PyTorch DataLoader worker count.")] = 4,
    model_name: Annotated[str, typer.Option(help="timm DINO/DINOv2 model name.")] = "vit_small_patch14_dinov2",
    image_size: Annotated[int, typer.Option(help="Image size used by transforms.")] = 512,
    device: Annotated[str, typer.Option(help="auto, cpu, cuda, or a torch device string.")] = "auto",
    augment: Annotated[bool, typer.Option(help="Apply train-only image augmentation.")] = True,
    rotation_degrees: Annotated[float, typer.Option(help="Maximum random train rotation in degrees.")] = 10.0,
    crop_scale_min: Annotated[float, typer.Option(help="Minimum random crop area fraction.")] = 0.9,
    crop_ratio_min: Annotated[float, typer.Option(help="Minimum random crop aspect ratio.")] = 0.95,
    crop_ratio_max: Annotated[float, typer.Option(help="Maximum random crop aspect ratio.")] = 1.05,
    blur_probability: Annotated[float, typer.Option(help="Probability of slight Gaussian blur.")] = 0.1,
    blur_kernel_size: Annotated[int, typer.Option(help="Odd Gaussian blur kernel size.")] = 3,
    jitter_probability: Annotated[float, typer.Option(help="Probability of mild brightness/contrast jitter.")] = 0.2,
    brightness_jitter: Annotated[float, typer.Option(help="Brightness jitter strength.")] = 0.08,
    contrast_jitter: Annotated[float, typer.Option(help="Contrast jitter strength.")] = 0.08,
) -> None:
    """Train a linear classifier on frozen DINO embeddings."""
    checkpoint_path = train_classifier(
        TrainingConfig(
            manifest_path=manifest,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
            num_workers=num_workers,
            model_name=model_name,
            image_size=image_size,
            device=device,
            augment=augment,
            rotation_degrees=rotation_degrees,
            crop_scale_min=crop_scale_min,
            crop_ratio_min=crop_ratio_min,
            crop_ratio_max=crop_ratio_max,
            blur_probability=blur_probability,
            blur_kernel_size=blur_kernel_size,
            jitter_probability=jitter_probability,
            brightness_jitter=brightness_jitter,
            contrast_jitter=contrast_jitter,
        )
    )
    typer.echo(f"wrote checkpoint: {checkpoint_path}")


def main() -> None:
    """Run the Typer application."""
    app()


if __name__ == "__main__":
    main()
