"""Command line interface for Parkinson drawing model training."""

from pathlib import Path
from typing import Annotated

import typer

from parkinson_model_train.config import TrainingConfig
from parkinson_model_train.data import labels_to_indices, load_manifest, split_by_subject, validate_image_paths
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
        )
    )
    typer.echo(f"wrote checkpoint: {checkpoint_path}")


def main() -> None:
    """Run the Typer application."""
    app()


if __name__ == "__main__":
    main()
