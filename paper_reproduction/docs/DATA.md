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
└── hemato/                                # cleaned moura-neto-2023-2025 release
    └── V<n>/{spiral,wave}/V<n>_<PARKINSON|CONTROL>_<SPIRAL|WAVE>_<group>_<idx>.png
```

A dataset that is absent is simply skipped when you build the manifest, so a
Kaggle+Folador-only `<data-root>` reproduces every non-Hemato cell.

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

### Expected manifest sizes (sanity check)

A correctly built manifest has exactly these counts. Subject counts are the
primary invariant (loader-independent); a mismatch means the data on disk is not
the cleaned release.

| dataset | subjects (HC / PD) | rows | spiral | wave | fold sizes |
|---|---|---|---|---|---|
| `kaggle`  | 28 (13 / 15) | 204  | 102  | 102  | 6, 6, 6, 5, 5 |
| `folador` | 40 (20 / 20) | 960  | 480  | 480  | 8, 8, 8, 8, 8 |
| `hemato`  | 46 (19 / 27) | **6990** | **3495** | **3495** | 10, 9, 9, 9, 9 |
| **total** | 114 (52 / 62) | 8154 | 4077 | 4077 | — |

For `hemato`, 6990 is the row count of the cleaned `moura-neto-2023-2025` image
tree (`hemato/V<n>/{spiral,wave}/*.png`), which the loader enumerates directly.
Every reported metric is subject-level (see [Labels](#labels)), so the exact
per-subject image count does not move any paper number — but this is the
authoritative row total for the released data.

## Labels

Binary, subject-level: `class_label ∈ {healthy, parkinson}`, with `parkinson`
as the positive class. The task is always the spiral drawing (`task_filter="spiral"`).

## A note on the released model

The released DINOv2 probe (see [`../models/MODEL_CARD.md`](../models/MODEL_CARD.md))
is fit on 100% of Hemato. Because Hemato is IRB-restricted, you must obtain it from
NIATS-UFU to re-run `models/export_hemato_probe.py` yourself. The serialized probe
on the Hugging Face Hub does not contain or expose any Hemato images.
