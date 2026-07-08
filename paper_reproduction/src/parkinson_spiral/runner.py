"""Split resolver + run_cell orchestrator + idempotency.

`run_cell(spec, ...)` is the only path that trains a model. It is idempotent: a
cell whose predictions already exist in `results/` is returned from cache, so a
re-run only fills in what is missing. Re-running the whole catalog therefore
converges to a complete local result directory without recomputing finished cells.
"""

import json
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .compute import ComputeMonitor
from .metrics import metrics_at_all_levels
from .protocol import CellSpec, Pipeline

_SUBJECT_CLEAN_CACHE: dict[tuple[str, int, float], pd.Series] = {}


def _task_mask(manifest: pd.DataFrame, task_filter: str) -> pd.Series:
    if task_filter == "both":
        return pd.Series(True, index=manifest.index)
    return manifest["drawing_type"] == task_filter


def resolve_split(manifest: pd.DataFrame, spec: CellSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_indices, test_indices) into `manifest` for `spec`."""
    spec.validate()
    task = _task_mask(manifest, spec.task_filter)

    if spec.split_source == "published":
        train = task & (manifest["dataset"] == spec.train_source) & (manifest["published_split"] == "train")
        test = task & (manifest["dataset"] == spec.test_source) & (manifest["published_split"] == "test")
        return manifest.index[train].to_numpy(), manifest.index[test].to_numpy()

    if spec.split_source == "cross-dataset":
        # Off-diagonal transfer: train on ALL of source A, test on ALL of source B. A and B are
        # disjoint subject populations, so no within-dataset split is needed.
        train = task & (manifest["dataset"] == spec.train_source)
        test = task & (manifest["dataset"] == spec.test_source)
        return manifest.index[train].to_numpy(), manifest.index[test].to_numpy()

    if spec.split_source == "subject-clean":
        train_split = _subject_clean_resplit(manifest, spec.train_source, spec.seed)
        test_split = _subject_clean_resplit(manifest, spec.test_source, spec.seed)
        train = (
            task
            & (manifest["dataset"] == spec.train_source)
            & (manifest.index.map(train_split.to_dict()).fillna("none") == "train")
        )
        test = (
            task
            & (manifest["dataset"] == spec.test_source)
            & (manifest.index.map(test_split.to_dict()).fillna("none") == "test")
        )
        return manifest.index[train].to_numpy(), manifest.index[test].to_numpy()

    # cv-fold: within-dataset subject CV. Train = all other folds of that dataset,
    # test = the held-out fold. Out-of-fold predictions are pooled across folds for
    # the diagonal ceiling.
    in_dataset = manifest["dataset"] == spec.train_source
    fold_match = manifest["cv_fold"] == spec.cv_fold
    train = task & in_dataset & ~fold_match & manifest["cv_fold"].notna()
    test = task & in_dataset & fold_match
    return manifest.index[train].to_numpy(), manifest.index[test].to_numpy()


def _subject_clean_resplit(manifest: pd.DataFrame, dataset: str, seed: int) -> pd.Series:
    key = (dataset, seed, 0.2)
    cached = _SUBJECT_CLEAN_CACHE.get(key)
    if cached is not None:
        return cached
    from .data.splits import produce_subject_clean_resplit

    series = produce_subject_clean_resplit(manifest, dataset, test_size=0.2, random_state=seed)
    _SUBJECT_CLEAN_CACHE[key] = series
    return series


# ---- run_cell ----


@dataclass
class RunPaths:
    manifest: Path
    outputs_dir: Path

    @property
    def predictions_dir(self) -> Path:
        return self.outputs_dir / "predictions"

    @property
    def cells_path(self) -> Path:
        return self.outputs_dir / "cells.parquet"


def _validate_manifest_paths(manifest: pd.DataFrame, *, side: str, sample: int) -> None:
    """Fail fast if a sample of image paths in `manifest` is unreachable."""
    if "file_path" not in manifest.columns:
        return
    paths = manifest["file_path"].astype(str).head(sample).tolist()
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"{len(missing)}/{len(paths)} sampled {side} image paths are unreachable "
            f"(e.g. {preview}). Check --data-root."
        )


def _check_train_classes(train_manifest: pd.DataFrame, spec: CellSpec) -> None:
    """Fail fast on configs that would otherwise train and emit plausible-looking numbers."""
    n = train_manifest["class_label"].nunique()
    if n < 2:
        raise ValueError(
            f"cell {spec.cell_id}: training set has {n} class(es) for a binary PD task; "
            "need both healthy and parkinson present in train."
        )


def _recipe_kwargs_for_spec(spec: CellSpec) -> dict[str, Any]:
    kwargs = dict(spec.recipe_kwargs)
    if spec.recipe_name == "AUG-Dogan":
        kwargs.setdefault("augment_then_split", spec.augment_then_split)
    return kwargs


def _configure_reproducibility(seed: int) -> None:
    """Seed the libraries shared by the paper pipelines."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _is_cached(paths: RunPaths, cell_id: str) -> bool:
    pred_path = paths.predictions_dir / f"{cell_id}.parquet"
    if not pred_path.is_file() or not paths.cells_path.is_file():
        return False
    cells = pd.read_parquet(paths.cells_path)
    return bool((cells["cell_id"] == cell_id).any())


def _append_cell_row(paths: RunPaths, row: dict[str, Any]) -> None:
    df = pd.DataFrame([row])
    if paths.cells_path.is_file():
        existing = pd.read_parquet(paths.cells_path)
        existing = existing[existing["cell_id"] != row["cell_id"]]
        df = pd.concat([existing, df], ignore_index=True)
    paths.cells_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths.cells_path, index=False)


def run_cell(spec: CellSpec, paths: RunPaths, *, force: bool = False) -> dict[str, Any]:
    """Train pipeline, predict, persist predictions parquet + cells.parquet row."""
    spec.validate()
    cell_id = spec.cell_id

    if not force and _is_cached(paths, cell_id):
        cells = pd.read_parquet(paths.cells_path)
        return cells.loc[cells["cell_id"] == cell_id].iloc[0].to_dict()

    # Lazy imports — keep the protocol surface importable without torch.
    from .data.dataset import PreprocessedSpiralDataset
    from .pipelines import get_pipeline
    from .recipes import get_recipe

    manifest = pd.read_parquet(paths.manifest)
    train_idx, test_idx = resolve_split(manifest, spec)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(f"empty split for cell {cell_id}: train={len(train_idx)}, test={len(test_idx)}")

    train_manifest = manifest.loc[train_idx].reset_index(drop=True)
    test_manifest = manifest.loc[test_idx].reset_index(drop=True)

    _check_train_classes(train_manifest, spec)
    _validate_manifest_paths(train_manifest, side="train", sample=5)
    _validate_manifest_paths(test_manifest, side="test", sample=5)

    recipe = get_recipe(spec.recipe_name, **_recipe_kwargs_for_spec(spec))
    train_ds = PreprocessedSpiralDataset(
        train_manifest,
        recipe,
        is_train=True,
        augment_then_split=spec.augment_then_split,
        target_column=spec.target_column,
    )
    test_ds = PreprocessedSpiralDataset(
        test_manifest,
        recipe,
        is_train=False,
        target_column=spec.target_column,
    )

    pipeline: Pipeline = get_pipeline(spec.pipeline_name, **spec.pipeline_kwargs)
    _configure_reproducibility(spec.seed)

    with ComputeMonitor() as fit_mon:
        pipeline.fit(train_ds, seed=spec.seed)

    with ComputeMonitor() as pred_mon:
        proba = pipeline.predict_proba(test_ds)

    y_true = (test_manifest["class_label"] == "parkinson").astype(int).to_numpy()
    pred_df = pd.DataFrame({
        "cell_id": cell_id,
        "image_id": test_manifest["image_id"].astype(str).to_numpy(),
        "subject_id": test_manifest["subject_id"].astype(str).to_numpy(),
        "session_id": test_manifest["session_id"].astype("string").to_numpy(),
        "dataset": test_manifest["dataset"].astype(str).to_numpy(),
        "drawing_type": test_manifest["drawing_type"].astype(str).to_numpy(),
        "y_true": y_true,
        "y_proba_pos": proba[:, 1],
    })

    paths.predictions_dir.mkdir(parents=True, exist_ok=True)
    pred_path = paths.predictions_dir / f"{cell_id}.parquet"
    pred_df.to_parquet(pred_path, index=False)

    metrics = metrics_at_all_levels(pred_df)
    n_test = len(test_idx)
    latency = (pred_mon.wall_seconds or 0.0) / max(n_test, 1) * 1000.0

    row: dict[str, Any] = {
        "cell_id": cell_id,
        "pipeline_name": spec.pipeline_name,
        "recipe_name": spec.recipe_name,
        "train_source": spec.train_source,
        "test_source": spec.test_source,
        "split_source": spec.split_source,
        "cv_fold": spec.cv_fold,
        "task_filter": spec.task_filter,
        "target_column": spec.target_column,
        "augment_then_split": spec.augment_then_split,
        "seed": spec.seed,
        "pipeline_kwargs": json.dumps(spec.pipeline_kwargs, sort_keys=True),
        "recipe_kwargs": json.dumps(spec.recipe_kwargs, sort_keys=True),
        "n_train_images": len(train_idx),
        "n_test_images": n_test,
        "n_train_subjects": int(train_manifest["subject_id"].nunique()),
        "n_test_subjects": int(test_manifest["subject_id"].nunique()),
        "fit_seconds": fit_mon.wall_seconds,
        "predict_seconds": pred_mon.wall_seconds,
        "gpu_peak_mb": fit_mon.gpu_peak_mb,
        "accelerator_device": getattr(pipeline, "_device", None),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "infer_latency_ms_per_img": latency,
        "predictions_path": str(pred_path),
    }
    for level, m in metrics.items():
        if m is None:
            continue
        for k, v in m.items():
            row[f"{level}_{k}"] = v
    _append_cell_row(paths, row)
    return row
