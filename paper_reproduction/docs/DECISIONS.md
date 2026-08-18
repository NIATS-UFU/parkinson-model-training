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

## Protocol Corrections vs Paper v5

Three defects were found during the full 97-cell run. All headline numbers in
this release differ from paper v5 as a result. The corrections are described
below with the exact code that was wrong and what replaced it.

### Bug 1 — Kaggle subject identity (kaggle.py)

**Root cause.** The original regex only matched spiral filenames:

```python
# BEFORE (wrong):
_FNAME_RE = re.compile(r"^(V\d+)([PH])E\d+\.png$", re.IGNORECASE)
```

The letter `E` (espiral) was hardcoded, so wave files (`O` = onda pattern,
e.g. `V01HO01.png`) did not match. For those files the fallback fired:

```python
base = Path(stem_with_ext).stem
return f"kaggle:{base}-{class_label[0].upper()}"
# → "kaggle:V01HO01-H" — one unique pseudo-ID per file
```

Every wave image received its own subject ID rather than being grouped with
its spiral counterpart. The manifest therefore contained ~130 "subjects"
(28 real spiral subjects + ~100 wave images as singletons) instead of the
correct 28.

Although `task_filter="spiral"` excluded wave images from training, fold
assignment ran on the full manifest before the filter. The 130-subject pool
changed which fold each real subject landed in compared to the correct 28-subject
pool, corrupting all in-domain CV splits for Kaggle.

**Fix.** Extend the regex to match both drawing tasks and common image extensions:

```python
# AFTER (correct):
_FNAME_RE = re.compile(r"^(V\d+)([PH])[EO]\d+\.(?:png|jpe?g)$", re.IGNORECASE)
```

All images for the same person now share a subject ID (`kaggle:V01H`,
`kaggle:V01P`). Deduplication yields the correct 28 subjects (13 healthy +
15 parkinson) before folds are assigned.

### Bug 2 — Image-row fold construction (splits.py)

**Root cause.** The original code called `StratifiedGroupKFold` directly on
all image rows, passing `subject_id` as the grouping key:

```python
# BEFORE (wrong):
sub = out.loc[mask].reset_index()
groups = sub["subject_id"].to_numpy()
y = (sub["class_label"] == "parkinson").astype(int).to_numpy()
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
splits = list(splitter.split(sub, y=y, groups=groups))
```

`StratifiedGroupKFold` distributes *image rows* (not subjects) across folds
while trying to keep groups intact. A Moura subject with 50 session images
counted 50× in the split objective; a Kaggle subject with a single image
counted 1×. Subjects with many images anchored individual folds, crowding out
smaller ones and producing unbalanced subject counts per fold. The class
stratification operated on image-count-weighted label distributions rather than
subject-level class ratios.

**Fix.** Extract a one-row-per-subject table, run `StratifiedKFold` on that,
then map fold numbers back to all image rows:

```python
# AFTER (correct):
rows = out.loc[mask, ["subject_id", "class_label"]].drop_duplicates()
subjects = rows.drop_duplicates("subject_id").reset_index(drop=True)
y = (subjects["class_label"] == "parkinson").astype(int).to_numpy()
splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
subject_to_fold: dict[str, int] = {}
for fold_idx, (_, val_idx) in enumerate(splitter.split(subjects, y=y)):
    for sid in subjects.iloc[val_idx]["subject_id"]:
        subject_to_fold[str(sid)] = fold_idx
out.loc[mask, "cv_fold"] = out.loc[mask, "subject_id"].map(subject_to_fold)
```

Each subject contributes exactly one vote to fold assignment regardless of how
many images they have. The 5-fold split is guaranteed class-stratified at the
subject level, consistent with the subject-level evaluation protocol.

### Bug 3 — Moura duplicate images (hemato.py)

**Root cause.** The Moura metadata parquet (`all_metadata.parquet`) was built
from session data collected under two naming conventions (V5 and V17 folders).
Twenty images appeared in entries for both naming schemes. The manifest builder
attempted to resolve each row's path but did not check whether the resolved
file actually existed on disk:

```python
# BEFORE (wrong): no existence check after path resolution
abs_path = (output_root / recorded).resolve()
manifest_rel = abs_path.relative_to(data_root.resolve()).as_posix()
rows.append({...})  # appended even if abs_path did not exist
```

Rows whose recorded path was stale (the session had been reorganised under the
other naming convention) were appended with a non-existent file path. Combined
with the rows for the same image under its active path, this produced 20 phantom
duplicate entries and an 8,178-row manifest instead of the correct 8,158.

If a pipeline attempted to load a phantom path, the image loader could return
zeros or trigger a silent skip, silently corrupting predictions.

**Fix.** Drop any row whose resolved path is not a real file:

```python
# AFTER (correct):
if not abs_path.is_file():
    continue
```

The manifest now has 8,158 rows, each backed by a file that exists. The 20
removed rows were split evenly across subjects, so no individual subject's fold
membership changed, but their absence corrects training-set composition in the
affected folds.

> **Counts above are as of the crop in hand when this fix landed.** They record what
> the fix did, not the size of any later dataset. Successive Moura crops and the
> `moura-neto-2023-2025` release layout each produce different totals, and
> `session_id` does not even mean the same thing across layouts. Fingerprint your own
> copy with `scripts/paper.py dataset-fingerprint` rather than comparing against these
> numbers — see [`DATA.md`](DATA.md).

## Experiment Semantics

- Cross-dataset cells with the same pipeline, recipe, seed, and training source
  reuse one fitted source model for both foreign test datasets. This preserves
  the intended matrix semantics: a matrix row is one trained source model
  evaluated against multiple targets, not two independently randomized fits.
- The Dogan paper reports augmentation before splitting, but does not publish the
  random split or variant assignment needed to reproduce that leakage. The
  released fidelity cell therefore uses the public released split and applies
  augmentation only to training images. The paper's 99.39% is retained only as
  an unverified, non-participant-disjoint upper bound.
- All three Dogan backbones are frozen public feature extractors in the bounded
  M4 reproduction. TensorFlow 2.21 exposes no Apple Metal device, so literal
  NASNetMobile fine-tuning would leave one branch on CPU and make the claimed
  three-backbone regime neither uniform nor submission-feasible.
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
