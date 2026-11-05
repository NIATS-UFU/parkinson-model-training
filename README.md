# Parkinson Model Training

[![on-merge](https://github.com/NIATS-UFU/parkinson-model-training/actions/workflows/on_merge.yaml/badge.svg)](https://github.com/NIATS-UFU/parkinson-model-training/actions/workflows/on_merge.yaml)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Python](https://img.shields.io/badge/python-3.13-blue.svg)

Training repo for the NIATS Parkinson drawing baseline: spiral/wave crops from [`spiral-cropper`](https://github.com/NIATS-UFU/spiral-cropper) go into a frozen DINO/DINOv2 encoder, then a supervised linear classifier is trained with subject-level splits.

**Looking for the paper's code?** *Cross-Dataset Robustness of Spiral Classifiers for Parkinson's
Disease Screening* is reproduced by [`paper_reproduction/`](paper_reproduction/README.md), a
self-contained package with its own environment, docs and licence.

- **Repository:** <https://github.com/NIATS-UFU/parkinson-model-training>
- **CLI entry point:** `main-cli`
- **Main assumption:** split by subject, not by image, to avoid leakage.
- **Training guide:** [`docs/training_walkthrough.md`](docs/training_walkthrough.md)
- **Colab notebook:** [`notebooks/colab_training.ipynb`](notebooks/colab_training.ipynb)

### Dataset names

The manuscript and this codebase name the same datasets differently. Documentation uses the paper's
name; commands and paths use the key verbatim.

| Key in code | Name in the paper |
|---|---|
| `kaggle` | Zham |
| `folador` | Folador |
| `hemato` | **Moura** — the NIATS-UFU clinical cohort, released as `moura-neto-2023-2025` |

`hemato` is a frozen identifier baked into published result files and the released model artifact; it
is never renamed. Read the cohort's name as *Moura* wherever the token appears.

## Quick Start

Install dependencies:

```sh
make install
```

Create or validate a manifest:

```sh
main-cli create-manifest hemato data/hemato data/manifests/hemato.csv
main-cli create-manifest niats-folador data/folador/HOG-IA-desenho-manuscrito/pyTremorDetect/dataset-junto data/manifests/niats_folador.csv
main-cli validate-manifest data/manifests/hemato.csv
main-cli split-summary data/manifests/hemato.csv --val-fraction 0.15 --test-fraction 0.15
```

Train the default DINOv2 baseline:

```sh
main-cli train \
  data/manifests/hemato.csv \
  runs/dinov2-hemato \
  --model-name vit_small_patch14_dinov2 \
  --image-size 518 \
  --epochs 20 \
  --batch-size 16
```

Read [`docs/training_walkthrough.md`](docs/training_walkthrough.md) for a notebook-style walkthrough with diagrams, plotting snippets, and practical checks before interpreting metrics.

For GPU training on Google Colab, open [`notebooks/colab_training.ipynb`](notebooks/colab_training.ipynb), select a GPU runtime, mount Google Drive, and edit the dataset path cell.

## Workflow

1. Produce standardized drawing crops with `spiral-cropper`.
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
main-cli train data/manifest.csv runs/dinov2-baseline --image-size 518 --epochs 20 --batch-size 16
```

Useful options:

- `--model-name`: any compatible `timm` model; default is `vit_small_patch14_dinov2`
- `--image-size`: input resize size. Use `518` for `vit_small_patch14_dinov2` with current timm builds.
- `--device`: `auto`, `cpu`, `cuda`, `mps`, or a Torch device string. On Apple Silicon, `auto` selects Metal Performance Shaders (MPS) when available.
- `--learning-rate`, `--weight-decay`: AdamW settings for the classifier head
- `--val-fraction`, `--test-fraction`, `--seed`: subject-level split controls
- `--num-workers`: PyTorch DataLoader worker count

Augmentation is applied to training images only, and is **on by default**. Pass `--no-augment` to
disable it entirely, or tune it with:

- `--rotation-degrees`: maximum random rotation, in degrees (default `10.0`)
- `--crop-scale-min`: minimum random-crop area fraction (default `0.9`)
- `--crop-ratio-min`, `--crop-ratio-max`: random-crop aspect-ratio bounds (defaults `0.95`, `1.05`)
- `--blur-probability`, `--blur-kernel-size`: slight Gaussian blur (defaults `0.1`, `3`; kernel must be odd)
- `--jitter-probability`, `--brightness-jitter`, `--contrast-jitter`: mild photometric jitter
  (defaults `0.2`, `0.08`, `0.08`)

The settings actually used are recorded in the checkpoint, so a run stays interpretable after the
fact.

Create manifests from local dataset layouts:

```sh
main-cli create-manifest hemato data/hemato data/manifests/hemato.csv
main-cli create-manifest niats-folador data/folador/HOG-IA-desenho-manuscrito/pyTremorDetect/dataset-junto data/manifests/niats_folador.csv
main-cli create-manifest image-tree data/my-labelled-images data/manifests/image_tree.csv
```

`hemato` loads the Moura cohort from `spiral-cropper` `5_final/` outputs, and is the primary
final-crop loader when you already have them.
`niats-folador` is the concise example for the Folador folder layout under `dataset-junto/` or `dataset-separado/`.
`image-tree` is the small generic fallback for labelled image folders when a custom manifest is not worth writing.

This root loader reads `**/5_final/*` crops only. The `moura-neto-2023-2025` de-identified release
ships a different tree and is read by the `paper_reproduction` loader instead — see
[`paper_reproduction/docs/DATA.md`](paper_reproduction/docs/DATA.md).

## Outputs

Training writes to the selected output directory:

- `best_classifier.pt`: classifier checkpoint, label mapping, image size, model name, augmentation settings, epoch, and validation metrics
- `run_summary.json`: split counts, subject counts, label mapping, and test metrics
- `test_metrics.svg`, `split_counts.svg`, `validation_history.svg`: standalone plots for quick inspection

## Modelling Rules

- Use NIATS `spiral-cropper` crops first.
- Split by `subject_id`, never by image row.
- Start with frozen DINO/DINOv2 embeddings before fine-tuning.
- Keep augmentation conservative for the first baseline.
- Treat inpainted or artifact-removed crops according to their metadata; do not silently mix them into raw-signal claims.
- Confirm the model input size before training. The default timm DINOv2 model expects `518x518` inputs.
- Check that validation and test splits contain both `PD` and `HC` subjects before interpreting accuracy.

## Reports And Plots

The main CLI writes a compact JSON summary and standalone SVG plots for every run:

```sh
python -m json.tool runs/dinov2-baseline/run_summary.json
```

Open `runs/dinov2-baseline/test_metrics.svg`, `split_counts.svg`, or `validation_history.svg` to inspect the run visually.

For epoch-by-epoch cross-validation curves, use the baseline script:

```sh
python train_baseline.py --data-dir data/hemato --model-type simple-cnn --epochs 5 --num-folds 5
```

This writes `runs/baseline_history.json` and the same standalone SVG plot files used by the main trainer.

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

## License

- **The paper's code** (`paper_reproduction/`): MIT, in
  [`paper_reproduction/LICENSE`](paper_reproduction/LICENSE).
- **Trained models:** CC-BY-NC-4.0 (attribution, non-commercial use only), stated on their
  [Zenodo record](https://doi.org/10.5281/zenodo.23748799).
