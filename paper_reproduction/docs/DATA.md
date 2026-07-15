# Datasets

This repository ships **no images** and no generated result parquet files. Layer-1
reproduction needs a local result directory. Layer-2 re-training needs the
datasets placed on disk in the layout below.

The paper uses three spiral-drawing datasets. **The dataset keys used in this code
differ from the names used in the paper** — this table is the mapping:

| Code key | Paper name | Public? | How to obtain |
|---|---|---|---|
| `kaggle` | **Zham** | Yes | Public Kaggle "Parkinson's Drawings" (Zham *et al.*) spiral/wave set. |
| `folador` | **Folador** | Yes | Folador *et al.* HOG-IA pen-and-paper handwriting dataset (`pyTremorDetect`). |
| `hemato` | **Moura** | **On request** | NIATS-UFU longitudinal clinical cohort (46 participants: 19 controls, 27 PD; spiral + wave), the de-identified `moura-neto-2023-2025` release. IRB-restricted; contact NIATS-UFU. |

The Moura dataset is the one the paper cites as `mouraneto2026dataset`; it appears
throughout the code and this document under its key, `hemato`.

## On-disk layout

Place the datasets under a single `<data-root>` directory:

```
<data-root>/
├── kaggle/
│   └── {spiral,wave}/{training,testing}/{healthy,parkinson}/*.png
├── folador/
│   └── HOG-IA-desenho-manuscrito/pyTremorDetect/dataset-junto/...
└── hemato/                                # de-identified moura-neto-2023-2025 release
    └── V<n>/{spiral,wave}/V<n>_<PARKINSON|CONTROL>_<SPIRAL|WAVE>_<drawing>_<sheet>.png
```

In the Hemato filename, the **second** number is the source **sheet** and the first is
which **drawing** on that sheet it is — so `V17_PARKINSON_SPIRAL_3_21` is drawing 3 of
sheet 21. Verified invariant across the release: **every sheet carries exactly 5
drawings** (1398 sheets x 5 = 6990 files). Drawing indices are `1..5` on 1383 sheets
and `2..6` on the remaining 15, so the first number ranges `1..6` even though the count
per sheet is always 5. Sheet counts vary by participant (`V2` has 13, `V17` has 22,
`V30` has 23).

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

> **Note — `build-manifest` drops images.** It applies `deduplicate_exact_images`
> (`data/unified.py`), which removes **byte-identical** files across the concatenated
> manifest. For the Hemato release that is 16 files (6990 → 6974), so the manifest
> will not match a `find | wc -l` of the tree. This is expected; the counts below are
> post-dedup. Note also that the byte hash does **not** find every duplicate — see
> [Duplicate sheets](#duplicate-sheets-and-why-dedup-does-not-remove-them-all).

### Expected manifest sizes (sanity check)

A correctly built manifest has exactly these counts. Subject counts are the
primary invariant (loader-independent); a mismatch means the data on disk is not
the cleaned release.

| dataset | subjects (HC / PD) | rows | spiral | wave | fold sizes |
|---|---|---|---|---|---|
| `kaggle`  | 28 (13 / 15) | 204  | 102  | 102  | 6, 6, 6, 5, 5 |
| `folador` | 40 (20 / 20) | 960  | 480  | 480  | 8, 8, 8, 8, 8 |
| `hemato`  | 46 (19 / 27) | **6974** | **3489** | **3485** | 10, 9, 9, 9, 9 |
| **total** | 114 (52 / 62) | 8138 | 4071 | 4067 | — |

For `hemato`, the cleaned `moura-neto-2023-2025` image tree
(`hemato/V<n>/{spiral,wave}/*.png`) holds **6990** files, and
`build_hemato_manifest` enumerates all of them. The manifest ends up with
**6974** because `build_unified_manifest` then applies
`deduplicate_exact_images` (`data/unified.py`), which drops 16 byte-identical
files. The table reports post-dedup counts, because that is what a built manifest
contains and what the audit in [`REPRODUCE.md`](REPRODUCE.md#audit-the-manifest-before-running)
prints.

### Duplicate sheets, and why dedup does not remove them all

File names are `V<n>_<LABEL>_<TYPE>_<drawing>_<sheet>.png`: the **second** number is
the source sheet, the first is which of the 5 drawings on that sheet it is. (So
`V2` has 13 sheets = 65 images per drawing type; `V30` has 23 = 115.)

**Sheets 21 and 22 are the same sheet, duplicated at source, for `V5` and `V17` — in
both spiral and wave.** That is 4 sheet pairs x 5 drawings = **20 duplicate images**.
Confirmed against the original sheets, and independently in pixels: after
normalising out translation and scale, matching drawings correlate at **+0.999**,
against **+0.02..+0.40** for different drawings by the same participant.

`deduplicate_exact_images` hashes **raw file bytes**, so it only removes the subset
that happens to be byte-identical:

| subject | type | sheets | duplicate images | dedup catches | dedup **misses** |
|---|---|---|---|---|---|
| `V5`  | spiral | 21 == 22 | 5 | 5 | 0 |
| `V5`  | wave   | 21 == 22 | 5 | 5 | 0 |
| `V17` | spiral | 21 == 22 | 5 | **0** | **5** |
| `V17` | wave   | 21 == 22 | 5 | 5 | 0 |

**`V17`'s spiral crops are the only ones the cropper framed slightly differently
between the two sheets** — same drawings, marginally different translation and
scale — so their bytes differ and all five survive dedup. A built manifest
therefore still contains **5 duplicate images**. Byte-identity is a property of how
a file was encoded and cropped, not of what it depicts, so the count dedup removes
should not be read as the amount of duplication present.

Separately, `V2`'s `SPIRAL_1_3` is byte-identical to `SPIRAL_3_1` (drawing 1 of
sheet 3 vs drawing 3 of sheet 1) — an index transposition rather than a duplicated
sheet. Dedup removes it. 15 sheet duplicates + this one = the 16 dropped.

Dedup does **not** affect fold assignment (folds are computed from a
one-row-per-subject table; see `data/splits.py`). Every reported metric is
subject-level (see [Labels](#labels)), so the 5 residual duplicates shift only
`V17`'s mean spiral prediction, and no paper number depends on the exact image
count.

## Labels

Binary, subject-level: `class_label ∈ {healthy, parkinson}`, with `parkinson`
as the positive class. The task is always the spiral drawing (`task_filter="spiral"`).

## A note on the released model

The released DINOv2 probe (see [`../models/MODEL_CARD.md`](../models/MODEL_CARD.md))
is fit on 100% of Hemato. Because Hemato is IRB-restricted, you must obtain it from
NIATS-UFU to re-run `models/export_hemato_probe.py` yourself. The serialized probe
on the Hugging Face Hub does not contain or expose any Hemato images.
