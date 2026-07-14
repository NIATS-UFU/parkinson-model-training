# Reproducing the paper

*"Cross-Dataset Robustness of Spiral Classifiers for Parkinson's Disease Diagnosis."*

Reproduction comes in **two layers**. Layer 1 needs a local result directory;
Layer 2 creates one from the datasets.

Use Python 3.13. The reproduction package pins its direct dependencies and ships
`requirements-lock.txt` for a fully pinned transitive environment.

| | What it does | Needs data? | Needs GPU? | Time |
|---|---|---|---|---|
| **Layer 1** | Rebuild every table and figure from local predictions | No | No | seconds |
| **Layer 2** | Re-train every model from scratch and regenerate the predictions | Yes | Some pipelines | hours |

Result parquet files are generated locally and are intentionally not committed to
the repository. A complete local result directory contains `cells.parquet` plus a
`predictions/` directory with one parquet file per experiment cell.

---

## Layer 1 — regenerate tables and figures from local results (no data, no GPU)

```bash
pip install -e .
python scripts/paper.py regenerate --results results --out paper_artifacts
```

For exact environment recreation, install from the lock first:

```bash
pip install -r requirements-lock.txt
pip install -e ".[train,hf]" --no-deps
```

This reads only the local result directory and writes:

- `paper_artifacts/results_tables.md` — the four paper tables (`b1-fidelity`,
  `cv-ceiling`, `cross-dataset`, `h4-cost`).
- `paper_artifacts/figures/tikz/fig1_crossdataset_matrices.tex`,
  `fig2_asymmetry.tex` — the TikZ figures.

The notebook (`notebooks/reproduce_experiments.ipynb`, Section 0) does the same
thing interactively after local results exist. Every number is computed from the
prediction parquet files by `src/parkinson_spiral/analysis.py` — nothing is hard-coded
except the published-literature reference values, which are clearly labelled.

---

## Layer 2 — re-train from scratch (needs data; some pipelines need a GPU)

Re-training produces `results/predictions/` from the raw images. It is
**idempotent**: each experiment is keyed by a `cell_id`, and a cell whose
predictions already exist is skipped, so an interrupted run resumes cleanly.

### 1. Get the data and build a manifest

See [`DATA.md`](DATA.md) for how to obtain each dataset and the on-disk layout.
Then:

```bash
pip install -e ".[train]"
python scripts/paper.py build-manifest --data-root <path-with-kaggle-folador-hemato> \
    --out results/manifest.parquet
```

A dataset that is absent is skipped by `scripts/paper.py reproduce`, so you can build a
Kaggle+Folador-only manifest and reproduce every cell that does not involve Hemato.

### Audit the manifest before running

A wrong manifest does **not** raise — it silently changes subject counts and fold
assignment. Confirm it against the expected sizes in [`DATA.md`](DATA.md#expected-manifest-sizes-sanity-check):

```bash
python - <<'PY'
import pandas as pd
m = pd.read_parquet("results/manifest.parquet")
subj = m[["dataset", "subject_id", "class_label"]].drop_duplicates()
print("subjects/dataset :", m.groupby("dataset")["subject_id"].nunique().to_dict())
print("subjects x class :", subj.groupby(["dataset", "class_label"]).size().to_dict())
print("rows/dataset     :", m.groupby("dataset").size().to_dict())
print("rows x drawing   :", m.groupby(["dataset", "drawing_type"]).size().to_dict())
folds = m[["dataset", "subject_id", "cv_fold"]].drop_duplicates().dropna(subset=["cv_fold"])
print("subjects/fold    :", folds.groupby(["dataset", "cv_fold"]).size().to_dict())
PY
```

Expected (must match exactly):

| dataset | subjects (HC / PD) | rows | fold sizes |
|---|---|---|---|
| `kaggle`  | 28 (13 / 15) | 204  | 6, 6, 6, 5, 5 |
| `folador` | 40 (20 / 20) | 960  | 8, 8, 8, 8, 8 |
| `hemato`  | 46 (19 / 27) | 6990 | 10, 9, 9, 9, 9 |

- **Kaggle ≈ 130 subjects** is the single most common failure: it means the wave-file
  (`O`) → participant mapping was not applied (see [`DECISIONS.md`](DECISIONS.md) Bug 1).
- The subject→fold map must equal [`../evidence/subject_folds.csv`](../evidence/subject_folds.csv)
  (114 subjects; Kaggle + Folador fold-identical).
- **Known repo cleanup:** the committed `runs/manifest.parquet` predates these fixes
  (it shows Kaggle 130) — regenerate it with the corrected builder or drop it, and
  reconcile the `checksums.sha256` / `results.zip` references so nobody builds off it.

### 2. Re-run the catalog

```bash
python scripts/paper.py reproduce --dry-run                       # list the 97 cells
python scripts/paper.py reproduce --manifest results/manifest.parquet --out results
python scripts/paper.py verify-catalog --results results          # cell_ids match the catalog
```

`--pipeline <name>` restricts the run to one pipeline (repeatable); `--force`
recomputes even cached cells.

### Which experiments need a GPU

| Pipeline | Training regime | GPU? |
|---|---|---|
| `dinov2_probe` | frozen public backbone → features → one logistic fit | **No** (CPU fine) |
| `folador_hog` | deterministic HOG features → SVM/RFC/KNN/CNN1D head | **No** |
| `dogan_triple` | three frozen public backbones + NCA + linear SVM | **Mixed** — PyTorch extraction uses MPS/CUDA; TensorFlow NASNetMobile may remain on CPU |
| `huang_vgg19` | VGG19 end-to-end fine-tune, 50 epochs | **Yes** |

CPU-only reproduction (DINOv2 + Folador HOG + Dogan-on-Hemato) covers the paper's
central H2/H4 story; the GPU pipelines are needed for the literature in-domain
ceilings and the full 3×3 matrices of Huang/Dogan.

On Apple Silicon, the PyTorch portions of DINOv2, Huang VGG19, and Dogan
automatically use the Metal Performance Shaders (`mps`) backend. CUDA mixed
precision remains disabled on MPS. Dogan's TensorFlow NASNetMobile branch uses
whatever TensorFlow device support is installed and may remain on CPU unless a
compatible TensorFlow Metal plugin is configured separately.

The notebook splits this for you: **Section 1** runs the CPU-only subset, and
**Section 2** runs the GPU pipelines behind a `torch.cuda.is_available()` guard.

---

## What "reproduces" means here

All reported numbers are **subject-level**, pooled across the cells that make up a
logical result (e.g. the 5 CV folds of a diagonal), with a percentile bootstrap
95% CI (`n_resamples=1000`, `seed=20260516`). Re-deriving them from generated
predictions is deterministic and exact once those predictions exist locally. One
small caveat about two GPU-trained cross-dataset AUCs is documented in
[`PROVENANCE.md`](PROVENANCE.md).
