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
`kaggle:V01P`). Deduplication yields the correct 28 subjects (14 healthy +
14 parkinson) before folds are assigned.

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
while trying to keep groups intact. A Hemato subject with 50 session images
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

### Bug 3 — Hemato duplicate images (hemato.py)

**Root cause.** Sessions 21 and 22 are duplicated at source, for both `V5` and
`V17`. Each pair holds the same five spiral and five wave drawings, byte-for-byte —
20 duplicate images:

```
output/V17/V17_PARKINSON_DADOS_DESENHOS_21/5_final/test1_sen_centered.png
output/V17/V17_PARKINSON_DADOS_DESENHOS_22/5_final/test1_sen_centered.png   # byte-identical
```

Both copies sit under the *same* naming convention. The export does use two
conventions (`RAW_DATA_drawings` for `V5`, `DADOS_DESENHOS` for `V17`), but that is
per-subject variation and not the source of the duplication.

The builder emits one row per metadata entry, so both copies were kept. Duplicates
overweight `V5` and `V17` — in training, and in their own subject-level mean
prediction. Folds are unaffected: `assign_subject_folds` reduces to one row per
subject, so per-subject image counts cannot reach fold construction.

**Fix.** `deduplicate_exact_images` (`data/unified.py`) hashes file bytes, rejects
copies whose identity columns disagree, and keeps the first:

```python
out.drop_duplicates("_content_sha256", keep="first")
```

This is what takes the manifest from 8,178 to 8,158 rows.

**Not fixed by the existence check.** An earlier version of this entry attributed
the 8,178 → 8,158 change to dropping rows whose resolved path is missing
(`if not abs_path.is_file(): continue`), and described the 20 as phantom entries
pointing at stale paths. That is wrong: against the released export every
`all_metadata.parquet` row resolves to a real file, so the existence check drops
**zero** rows. The images are real duplicates and the whole delta is deduplication.

**Still open under the image-tree loader.** The cleaned `moura-neto-2023-2025`
release carries the same duplication as sheets 21 and 22, but re-cropped `V17`'s
spirals slightly, so only **15 of the 20** remain byte-identical. Dedup removes
those 15; the 5 `V17` spiral duplicates survive into the manifest. Byte-identity
tracks how a file was encoded, not what it depicts, so the number dedup drops is not
the amount of duplication present. See
[`DATA.md`](DATA.md#duplicate-sheets-and-why-dedup-does-not-remove-them-all).

The manifest now has 8,158 rows, each backed by a file that exists. The 20
removed rows were split evenly across subjects, so no individual subject's fold
membership changed, but their absence corrects training-set composition in the
affected folds.

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
