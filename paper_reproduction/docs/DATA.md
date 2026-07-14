# Datasets

This repository ships **no images** and no generated result parquet files. Layer-1
reproduction needs a local result directory. Layer-2 re-training needs the
datasets placed on disk in the layout below.

The paper uses three spiral-drawing datasets:

| Name (paper) | Public? | How to obtain |
|---|---|---|
| `kaggle` | Yes | Public Kaggle "Parkinson's Drawings" (Zham et al.) spiral/wave set. |
| `folador` | Yes | Folador *et al.* HOG-IA pen-and-paper handwriting dataset (`pyTremorDetect`). |
| `hemato` | **On request** | NIATS-UFU clinical cohort. IRB-restricted; available on request — contact NIATS-UFU. |

## On-disk layout

Place the datasets under a single `<data-root>` directory:

```
<data-root>/
├── kaggle/
│   └── {spiral,wave}/{training,testing}/{healthy,parkinson}/*.png
├── folador/
│   └── HOG-IA-desenho-manuscrito/pyTremorDetect/dataset-junto/...
└── hemato/
    └── output/
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

Session folders whose name does not match `V<n>_(PARKINSON|CONTROLE)_` are
silently ignored. Each `5_final/` directory holds exactly the `*_centered.png`
crops produced by spiral-cropper; there is no central parquet in `output/`.

A dataset that is absent is simply skipped when you build the manifest, so a
Kaggle+Folador-only `<data-root>` reproduces every non-Hemato cell.

### Legacy Hemato layout (all_metadata.parquet)

If you have the older distribution (a single `all_metadata.parquet` file plus a
`V<n>/V<n>_*/5_final/` nested structure), place it at:

```
<data-root>/hemato/output/all_metadata.parquet
```

`build_hemato_manifest` detects the parquet and uses the legacy loader
automatically. An `output_v2/` directory is also supported for backward
compatibility with the previous symlink name.

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

### Kaggle — 28 subjects (14 healthy + 14 parkinson)

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
class-ratio drift for datasets like Hemato where session counts vary widely per subject.

### Hemato — only existing image files

The manifest builder resolves every `final_centered_rel_path` entry to an
absolute path and drops rows where `abs_path.is_file()` is False. The flat-output
scanner also only writes a row for files that are actually present on disk. This
removes stale/phantom entries that arise when session folders are reorganised between
spiral-cropper runs. A clean `output/` manifest should contain:

- **698 sessions** across all subjects
- **6 980 image rows** (exactly 10 per session: 5 spiral + 5 wave)

If your local count differs materially, check for missing `5_final/` directories
or extra session folders that do not match the naming pattern.

## Labels

Binary, subject-level: `class_label ∈ {healthy, parkinson}`, with `parkinson`
as the positive class. The task is always the spiral drawing (`task_filter="spiral"`).

## A note on the released model

The released DINOv2 probe (see [`../models/MODEL_CARD.md`](../models/MODEL_CARD.md))
is fit on 100% of Hemato. Because Hemato is IRB-restricted, you must obtain it from
NIATS-UFU to re-run `models/export_hemato_probe.py` yourself. The serialized probe
on the Hugging Face Hub does not contain or expose any Hemato images.
