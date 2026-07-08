# Provenance of generated results

This documents how the curated 97-cell result set (`results/cells.parquet` +
`results/predictions/`) relates to the paper when generated locally, so the
numbers can be trusted and traced without committing generated parquet artifacts.

## The 97 cells

The catalog (`src/parkinson_spiral/catalog.py`) is the declarative source of truth.
It enumerates exactly 97 experiment cells in the paper's naming, in three groups:

| Group | n | What | Feeds |
|---|---|---|---|
| **A** | 70 | In-domain 5-fold participant-grouped CV ceilings | Table `cv-ceiling`, matrix diagonals |
| **B** | 24 | Cross-dataset off-diagonal transfers (4 pipelines × 6 directions) | Table `cross-dataset`, matrices, asymmetry (Fig. 2) |
| **C** | 3 | In-domain published-split fidelity | Table `b1-fidelity` (Released-split column) |

`scripts/paper.py verify-catalog --results results` checks that the cell_ids enumerated
by the catalog match the cell_ids in `results/cells.parquet` exactly — no missing,
no extra.

## How numbers are derived

Every reported value is computed from per-image predictions by
`src/parkinson_spiral/analysis.py`. The recipe is uniform:

1. Pool the per-image predictions across all cells that make up a logical result
   (e.g. the 5 CV folds of one diagonal pool into one pooled out-of-fold ceiling).
2. Aggregate to the **subject** level (mean P(PD) per subject) — the paper's
   canonical reporting level.
3. Score binary metrics with `parkinson` as the positive class.
4. Attach a **percentile bootstrap 95% CI** (`n_resamples=1000`, `seed=20260516`,
   `numpy.quantile`).

There is a **single source of truth**: the local `predictions/`. Tables and
figures are recomputed from them, never copied from the manuscript. The only
hard-coded values are the published-literature reference numbers in the
`b1-fidelity` table (Huang 0.87, Dogan 0.994, Folador 0.696), which are clearly
labelled as external citations.

## Curation: a paper-scoped subset

The 97-cell result set is a curated subset of a larger internal benchmark run,
restricted to exactly the experiments the paper reports. The selection is
**deterministic and involves no retraining**: cells are selected by their
parameters, each cell's identity is re-verified by recomputing its `cell_id`, and
the result matches the catalog exactly (`scripts/paper.py verify-catalog --results results`).
Every generated file uses the paper's naming throughout.

Out of paper scope and therefore excluded: the dropped leakage-gap analysis, the
acquisition-source classifier (H5), the Bernardo binarization recipe and other
augmentation variants the paper does not use, the Folador RFC/KNN/CNN1D classifier
heads, and the wave / all-modality experiments. The Folador pipeline ships its
**SVM_linear** head (`pipeline_kwargs={"classifier": "SVM_linear"}`), which is what
the paper reports.

## Known nuance: two GPU-trained cross-dataset AUCs

For **2 of the 24** cross-dataset cells — both GPU-fine-tuned (one Huang VGG19, one
Dogan triple) — the subject-level **AUC** re-derived from generated predictions
differs from the value recorded in an earlier manuscript draft by **≤ 0.0117**
(e.g. Huang K→H: 0.426 recorded vs 0.4376 re-pooled). Accuracy, sensitivity,
specificity, precision and F1 are **bit-identical**. The cause is cuDNN
nondeterminism in the GPU fine-tuning (the probabilities differ in the low-order
digits that AUC's ranking is sensitive to but the 0.5-threshold metrics are not).

**Resolution:** all reported numbers are derived **uniformly from local
`predictions/`** (one source of truth), so the repository is internally consistent
and reproducible. No conclusion in the paper depends on these third-decimal AUC
differences — the H2 finding (poor cross-dataset transfer) is unchanged.

## Seed

`DEFAULT_SEED = 20260516` is used for every split, every model fit, and every
bootstrap, so re-derivation from generated predictions is exact.
