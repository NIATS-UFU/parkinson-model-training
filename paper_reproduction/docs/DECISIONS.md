# PR Details and Decisions

This file records the engineering decisions made while hardening the paper
reproduction release for review.

## Runtime

- Python is standardized on **3.13** for both the root package and the paper
  reproduction package (`>=3.13,<3.14`).
- `from __future__ import annotations` imports are intentionally removed because
  Python 3.13 is the minimum runtime.
- The paper reproduction package caps Python at `<3.14` until the data-loading
  stack and worker-start behavior are validated under Python 3.14+.

## Reproducibility

- Layer-1 dependencies are pinned in `pyproject.toml` and `requirements.txt` so
  table and figure regeneration uses a stable numerical stack.
- `requirements-lock.txt` records the complete transitive environment for exact
  Python 3.13 reruns, including Layer-2 training and Hugging Face publishing extras.
- The DINOv2 `torch.hub` download is pinned to upstream commit
  `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; torchvision, timm, and TensorFlow/Keras
  pretrained loaders are controlled by the locked package versions.
- Layer-2 training extras are pinned to avoid silent changes in framework,
  pretrained-weight, and augmentation behavior.
- `scripts/paper.py verify-catalog` validates both the catalog cell IDs and the
  prediction parquet files for presence, schema, non-empty contents, and matching
  `cell_id` values.
- `scripts/paper.py reproduce` skips cells whose train/test datasets are absent from the
  manifest, making partial public-data reproductions explicit instead of failing
  midway through the catalog.

## Experiment Semantics

- The Dogan published-split fidelity cell propagates `augment_then_split=True` into
  the `AUG-Dogan` recipe so the dataset emits the intended eight deterministic
  rotation/flip variants per source image.
- The catalog remains the source of truth for the 97 full-paper cells. Partial runs
  are allowed only as local reruns when the manifest lacks restricted data.

## Tests and CI

- Root `make test` now runs the root test suite and the `paper_reproduction/tests`
  suite in the reproduction package environment.
- Reproduction tests remain allowed to skip result-dependent assertions when local
  generated parquet outputs are absent.

## Code Structure

- Small helpers were kept where they encode domain concepts, such as catalog group
  construction and metric aggregation.
- Behavioral review findings were fixed with targeted changes instead of broad
  module restructuring, preserving the paper-facing API and experiment catalog.
- The paper reproduction entry points are consolidated in `scripts/paper.py` to
  avoid five small scripts that split one logical CLI across multiple files.
