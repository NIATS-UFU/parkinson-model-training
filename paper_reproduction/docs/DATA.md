# Datasets

This repository ships **no images** and no generated result parquet files. Layer-1
reproduction needs a local result directory. Layer-2 re-training needs the
datasets placed on disk in the layout below.

The paper uses three spiral-drawing datasets. The paper and the code name them
differently; the code key is frozen and is what every command and result file uses.

| Name in the paper | Key in code | Public? | How to obtain |
|---|---|---|---|
| Zham | `kaggle` | Yes | Public Kaggle "Parkinson's Drawings" (Zham et al.) spiral/wave set. |
| Folador | `folador` | Yes | Folador *et al.* HOG-IA pen-and-paper handwriting dataset (`pyTremorDetect`). |
| **Moura** | `hemato` | **On request** | NIATS-UFU clinical cohort, released as `moura-neto-2023-2025`. IRB-restricted; available on request — contact NIATS-UFU. |

Read `hemato` as *Moura* throughout: the key is hashed into `cell_id`, `image_id`
and `subject_id`, so renaming it would invalidate the published results.

## On-disk layout

Place the datasets under a single `<data-root>` directory:

```
<data-root>/
├── kaggle/
│   └── {spiral,wave}/{training,testing}/{healthy,parkinson}/*.png
├── folador/
│   └── HOG-IA-desenho-manuscrito/pyTremorDetect/dataset-junto/...
└── hemato/
    ├── V<n>/spiral/V<n>_<PARKINSON|CONTROL>_SPIRAL_<drawing>_<sheet>.png
    └── V<n>/wave/  V<n>_<PARKINSON|CONTROL>_WAVE_<drawing>_<sheet>.png
```

The Moura block above is the **de-identified `moura-neto-2023-2025` release** — the
form a collaborator receives, and the layout the published run read. `subject_id`
comes from the `V<n>` folder, drawing type from the subfolder, and the class label
from the filename token.

A filename that does not match that pattern **raises** rather than being skipped, so
a mis-packed release fails loudly instead of silently producing a short manifest.
The per-session dedup of the cropper layouts does not apply to the release tree. The
release does hold 18 byte-identical pairs: sheets 21 and 22 of V5 and of V17 are one
scan filed twice. The global exact-image dedup below keeps one of each, so its 6,990
files give 6,972 Moura rows.

A dataset that is absent is simply skipped when you build the manifest, so a
Zham+Folador-only `<data-root>` reproduces every cell that does not involve Moura.

### Other supported Moura layouts

`build_hemato_manifest` selects a layout from what is on disk, in this order — the
**first match wins**, so an `output/` directory takes precedence over the release
tree:

1. `hemato/output/` **without** `all_metadata.parquet` → flat spiral-cropper session scan
2. `hemato/output/` **with** `all_metadata.parquet` → legacy parquet scan
3. `hemato/output_v2/` → same flat scan, under the old symlink name
4. `hemato/V<n>/{spiral,wave}/*.png` → the release image tree documented above
5. none of the above → an empty manifest, and every Moura cell is skipped

The spiral-cropper tree (1 and 3) looks like:

```
<data-root>/hemato/output/
├── V<n>_PARKINSON_RAW_DATA_drawings_<N>/
│   └── 5_final/
│       ├── test1_spi_centered.png
│       ├── test1_sen_centered.png
│       └── ...
├── V<n>_PARKINSON_DADOS_DESENHOS_<N>/
│   └── 5_final/ ...
└── V<n>_CONTROLE_RAW_DATA_drawings_<N>/
    └── 5_final/ ...
```

Here `subject_id` is the `V<n>` prefix and the drawing type comes from the filename
(`*_spi_*` → spiral, `*_sen_*` → wave). Note the **opposite** error policy to the
release tree: session folders whose name does not match `V<n>_(PARKINSON|CONTROLE)_`
are *silently ignored*.

For the legacy distribution (2), place the single metadata file at
`<data-root>/hemato/output/all_metadata.parquet`; the loader detects it and joins
`subject_id` and `class_label` automatically.

### Deduplication

Two mechanisms remove duplicate images, and both can make your row count differ from
a naive file count:

- **Per-subject session dedup** (cropper layouts only): a session whose `5_final/`
  contents are byte-identical to an earlier session for the same subject is skipped
  with a warning. This handles the same sheets being re-exported under two naming
  conventions.
- **Global exact-image dedup** (all datasets): byte-identical images are collapsed
  across the whole manifest. If two identical images carry *conflicting* metadata,
  this raises rather than picking one.

## Building the manifest

```bash
python scripts/paper.py build-manifest --data-root <data-root> --out results/manifest.parquet
```

This writes one row per image with a unified schema (`dataset`, `subject_id`,
`image_id`, `class_label`, `drawing_type`, `file_path`, …) plus a 5-fold
**participant-grouped** `cv_fold`. Splits are always subject-level: no subject
appears in both train and test of a CV fold. Cross-dataset cells train on **all**
of one dataset and test on **all** of another (the three cohorts are disjoint
populations).

## Manifest invariants and cleaning steps

These invariants hold for the corrected manifest used in this release. A coworker
re-building the manifest should verify them before running the full catalog.
Detailed rationale for each is in [`DECISIONS.md`](DECISIONS.md).

### Zham (`kaggle`) — 28 subjects (13 healthy + 15 parkinson)

Wave filenames (`V<n>H/PO<d>.png`) must map to the same subject ID as their
spiral counterpart (`V<n>H/PE<d>.png`). The character after the class letter is
`E` (espiral) or `O` (onda), not a stable distinguisher. The manifest builder
normalises both to `kaggle:V<n>H` / `kaggle:V<n>P`.

Verify: after `build-manifest`, `manifest["subject_id"].str.startswith("kaggle:")` rows
grouped by `subject_id` should yield exactly 28 unique subjects.

### Folds — subject-level stratification

Cross-validation folds are assigned by `StratifiedKFold` on a **one-row-per-subject**
table, then mapped back to all image rows. Each subject receives exactly one fold
regardless of how many images they have. Image-weighted splitting (the pre-correction
approach with `StratifiedGroupKFold` on image rows) produced unbalanced folds and
class-ratio drift for datasets like Moura where session counts vary widely per subject.

### Moura (`hemato`) — only existing image files

The manifest builder resolves every `final_centered_rel_path` entry to an
absolute path and drops rows where `abs_path.is_file()` is False. The flat-output
scanner also only writes a row for files that are actually present on disk. This
removes stale/phantom entries that arise when session folders are reorganised between
spiral-cropper runs.

**Row and session counts are not fixed, so this document does not state one.** They
depend on which crop you hold and which of the layouts above you use, and successive
crops have legitimately produced different totals. Fingerprint your own copy instead:

```bash
python scripts/paper.py dataset-fingerprint --manifest results/manifest.parquet
```

Compare the result against the run you are trying to reproduce, rather than against a
number written here. A collaborator who already has a reference fingerprint can check
a copy byte-for-byte with `verify-fingerprint`.

One caveat when comparing across layouts: `session_id` does not mean the same thing in
both. The cropper layouts derive it from the **session folder**, one per physical
sheet; the release tree sets it to `V<n>_<drawing>`, grouping by **position on the
sheet**. Session counts from the two layouts are therefore not comparable. Every
reported metric is subject-level, so nothing published depends on this.

## Labels

Binary, subject-level: `class_label ∈ {healthy, parkinson}`, with `parkinson`
as the positive class. The task is always the spiral drawing (`task_filter="spiral"`).

## A note on the trained models

The four published models are fit on all of Moura. No model file holds an image or a
single drawing's features: the DINOv2 probe is a logistic regression, the two linear
SVMs are stored as their weight vector, Dogan's NCA as its projection matrix, and Huang
VGG19 as its network weights. Rebuilding them needs Moura, on request from NIATS-UFU.
