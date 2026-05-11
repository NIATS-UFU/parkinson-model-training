# Parkinson Model Train

[![on-merge](https://github.com/NIATS-UFU/parkinson-model-train/actions/workflows/on_merge.yaml/badge.svg)](https://github.com/NIATS-UFU/parkinson-model-train/actions/workflows/on_merge.yaml)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)

Training repo for the NIATS Parkinson drawing baseline: 512x512 spiral/wave crops from [`spiral-cropper`](https://github.com/NIATS-UFU/spiral-cropper) go into a frozen DINO/DINOv2 encoder, then a supervised linear classifier is trained with subject-level splits.

- **Repository:** <https://github.com/NIATS-UFU/parkinson-model-train>
- **Local path:** `/data/git/ufu/parkinson_model_training`
- **CLI entry point:** `main-cli`
- **Main assumption:** split by subject, not by image, to avoid leakage.

## Workflow

1. Produce standardized 512x512 drawing crops with `spiral-cropper`.
2. Create a CSV manifest with image paths, subject IDs, and labels.
3. Validate the manifest and preview the subject-level split.
4. Train the frozen DINO classifier baseline.
5. Review `run_summary.json` before interpreting model metrics.

## Installation

```sh
make install
```

This installs the Python package and development dependencies with `uv`. The project currently uses PyTorch CUDA 12.6 wheels on Linux.

## Manifest

The training CLI expects a CSV manifest. Relative image paths are resolved from the manifest directory.

```csv
image_path,subject_id,label
5_final/p001_spi_01.png,p001,PD
5_final/c014_spi_01.png,c014,HC
```

Required columns:

- `image_path`: path to a crop image, usually from `spiral-cropper` `5_final/`
- `subject_id`: participant identifier used for leakage-safe splitting
- `label`: class label, for example `PD` or `HC`

## Commands

Validate the manifest and image paths:

```sh
main-cli validate-manifest data/manifest.csv
```

Preview train/validation/test sizes using subject-level splits:

```sh
main-cli split-summary data/manifest.csv --val-fraction 0.15 --test-fraction 0.15
```

Train the baseline:

```sh
main-cli train data/manifest.csv runs/dinov2-baseline --epochs 20 --batch-size 16
```

Useful options:

- `--model-name`: any compatible `timm` model; default is `vit_small_patch14_dinov2`
- `--image-size`: input resize size; default is `512`
- `--device`: `auto`, `cpu`, `cuda`, or a Torch device string
- `--learning-rate`: AdamW learning rate for the classifier head
- `--val-fraction`, `--test-fraction`, `--seed`: subject-level split controls
- `--num-workers`: PyTorch DataLoader worker count

## Outputs

Training writes to the selected output directory:

- `best_classifier.pt`: classifier checkpoint, label mapping, image size, model name, and validation metrics
- `run_summary.json`: split counts, subject counts, label mapping, and test metrics

## Modelling Rules

- Use NIATS `spiral-cropper` 512x512 crops first.
- Split by `subject_id`, never by image row.
- Start with frozen DINO/DINOv2 embeddings before fine-tuning.
- Keep augmentation conservative for the first baseline.
- Treat inpainted or artifact-removed crops according to their metadata; do not silently mix them into raw-signal claims.

## Development

Run the full local check before presenting changes:

```sh
make check
```

Run only tests:

```sh
make test
```

Format and lint:

```sh
make fmt
make lint
```
