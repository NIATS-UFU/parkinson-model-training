"""Subject-level splits: GroupKFold for the within-dataset diagonal, published
passthrough, and a subject-clean re-split helper.

Decisions:
- n_splits = 5.
- random_state = 20260516 (DEFAULT_SEED).
- StratifiedGroupKFold (sklearn >=1.1) for class balance across folds; falls back
  to GroupKFold with a warning if unavailable.
- Folds are assigned PER DATASET (each dataset folded over its own subjects), so
  the in-domain ceiling is a within-dataset CV mean for every dataset — not a
  single high-variance subject-clean draw.
"""

from __future__ import annotations

import warnings

import pandas as pd

from .. import DEFAULT_SEED

CV_DATASETS = ("kaggle", "folador", "hemato")


def published_train_mask(manifest: pd.DataFrame, dataset: str) -> pd.Series:
    return (manifest["dataset"] == dataset) & (manifest["published_split"] == "train")


def published_test_mask(manifest: pd.DataFrame, dataset: str) -> pd.Series:
    return (manifest["dataset"] == dataset) & (manifest["published_split"] == "test")


def _assign_one_dataset(out: pd.DataFrame, dataset: str, n_splits: int, random_state: int) -> None:
    """In-place: fill `cv_fold` for one dataset's rows via subject-stratified KFold."""
    mask = out["dataset"] == dataset
    if not mask.any():
        return
    sub = out.loc[mask].reset_index()  # keeps original index in 'index'
    groups = sub["subject_id"].to_numpy()
    y = (sub["class_label"] == "parkinson").astype(int).to_numpy()

    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(splitter.split(sub, y=y, groups=groups))
    except (ImportError, TypeError):
        warnings.warn(
            "StratifiedGroupKFold unavailable; falling back to GroupKFold (no class stratification)",
            stacklevel=2,
        )
        from sklearn.model_selection import GroupKFold

        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(sub, y=y, groups=groups))

    fold_col = pd.Series(pd.NA, index=sub.index, dtype="Int64")
    for fold_idx, (_train_idx, val_idx) in enumerate(splits):
        fold_col.iloc[val_idx] = fold_idx

    out.loc[mask, "cv_fold"] = fold_col.set_axis(sub["index"]).reindex(out.index[mask]).values


def assign_subject_folds(
    manifest: pd.DataFrame,
    datasets: tuple[str, ...] = CV_DATASETS,
    n_splits: int = 5,
    random_state: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Return a copy of `manifest` with `cv_fold` filled in per dataset.

    Each dataset is folded independently over its own subjects, so the result for
    any one dataset depends only on that dataset's rows + the seed."""
    out = manifest.copy()
    for ds in datasets:
        _assign_one_dataset(out, ds, n_splits=n_splits, random_state=random_state)
    return out


def produce_subject_clean_resplit(
    manifest: pd.DataFrame,
    dataset: str,
    test_size: float = 0.2,
    random_state: int = DEFAULT_SEED,
) -> pd.Series:
    """Return a 'train'/'test' Series for one dataset, derived with GroupShuffleSplit
    so no subject crosses train/test."""
    from sklearn.model_selection import GroupShuffleSplit

    sub = manifest.loc[manifest["dataset"] == dataset]
    if sub.empty:
        return pd.Series(dtype="string")
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    ((train_idx, test_idx),) = gss.split(sub, groups=sub["subject_id"].to_numpy())
    out = pd.Series(index=sub.index, dtype="string")
    out.iloc[train_idx] = "train"
    out.iloc[test_idx] = "test"
    return out
