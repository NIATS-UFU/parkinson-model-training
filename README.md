# Parkinson Model Train

[![on-merge](https://github.com/NIATS-UFU/parkinson-model-train/actions/workflows/on_merge.yaml/badge.svg)](https://github.com/NIATS-UFU/parkinson-model-train/actions/workflows/on_merge.yaml)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Python](https://img.shields.io/badge/python-3.13-blue.svg)

Training repo for the NIATS Parkinson drawing baseline: 512x512 spiral/wave crops from [`spiral-cropper`](https://github.com/NIATS-UFU/spiral-cropper) go into a frozen DINO/DINOv2 encoder, then a supervised linear classifier is trained with subject-level splits.

- **Repository:** <https://github.com/NIATS-UFU/parkinson-model-train>
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

This installs the Python package and development dependencies with `uv`. The
project currently uses Python 3.13 and PyTorch CUDA 12.6 wheels on Linux.
The committed `uv.lock` is the source of truth for the root environment; the
paper reproduction package has its own committed lockfile and Python 3.13
environment.

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
- `--device`: `auto`, `cpu`, `cuda`, `mps`, or a Torch device string. On Apple
  Silicon, `auto` selects Metal Performance Shaders (MPS) when available.
- `--learning-rate`: AdamW learning rate for the classifier head
- `--val-fraction`, `--test-fraction`, `--seed`: subject-level split controls
- `--num-workers`: PyTorch DataLoader worker count

Create manifests from local dataset layouts:

```sh
main-cli create-manifest handpd data/handpd data/manifests/handpd.csv
main-cli create-manifest newhandpd data/newhandpd data/manifests/newhandpd.csv
main-cli create-manifest hemato data/hemato data/manifests/hemato.csv
main-cli create-manifest kaggle-drawings data/kaggle/parkinsons-drawings data/manifests/kaggle_drawings.csv
main-cli create-manifest kaggle-spiral data/kaggle_2 data/manifests/kaggle_2.csv
main-cli create-manifest roboflow data/roboflow data/manifests/roboflow.csv
main-cli create-manifest folador data/folador data/manifests/folador.csv
```

`kaggle-drawings` targets Kaggle `kmader/parkinsons-drawings` and reads nested spiral/wave train/test class folders. The dataset does not expose reliable participant IDs, so the loader uses image-level pseudo-subject IDs; use it only for smoke tests or literature-comparison baselines, not as primary evidence.

The NIATS cropper metadata at `/data/git/ufu/crop_spiral_tests/output/all_metadata.parquet` still needs to be joined with `subject_id` and clinical `label` before it can be used for the primary benchmark.

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
