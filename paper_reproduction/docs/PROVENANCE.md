# Provenance of generated results

This documents how a 97-cell result set (`results/cells.parquet` +
`results/predictions/`), generated locally, relates to the paper, so the
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

## Scope of the catalog

The catalog holds exactly the experiments the paper reports, in the paper's naming. A
result set is produced by running `scripts/paper.py reproduce` over it, and
`scripts/paper.py verify-catalog --results results` confirms that the result set and
the catalog name the same 97 cells. The Folador pipeline runs its **SVM_linear** head
(`pipeline_kwargs={"classifier": "SVM_linear"}`), which is what the paper reports.

## The commit the frozen evidence names

`evidence/environment.json` records commit `8d2aed1`, the commit that froze the evidence. The
97 results it describes were produced at `e18573a`. Only the scripts, tests and documentation
changed between the two: the `parkinson_spiral` package is byte-identical, so both give the code
signature `e112a2ad3213…` that every result cell records.

## Seed

`DEFAULT_SEED = 20260516` is used for every split, every model fit, and every
bootstrap, so re-derivation from generated predictions is exact.
