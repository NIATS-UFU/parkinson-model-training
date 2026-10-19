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
Zham+Folador-only manifest and reproduce every cell that does not involve Moura (the
`hemato` key).

To confirm your copy of the data is the one a given run used:

```bash
python scripts/paper.py dataset-fingerprint --manifest results/manifest.parquet
python scripts/paper.py verify-fingerprint --manifest results/manifest.parquet
```

`dataset-fingerprint` hashes every image and writes `evidence/dataset_fingerprint.csv`
plus a summary; `verify-fingerprint` checks a local copy against a reference
fingerprint byte-for-byte. Run these before a long reproduction rather than after.

### 2. Re-run the catalog

```bash
python scripts/paper.py reproduce --dry-run                       # list the 97 cells
python scripts/paper.py reproduce --manifest results/manifest.parquet --out results
python scripts/paper.py verify-catalog --results results          # cell_ids match the catalog
```

> **`reproduce --out` defaults to `runs/reproduce`, not `results/`.** Every example
> here passes `--out results` explicitly. If you omit it, predictions land in
> `runs/reproduce` and a later `verify-catalog --results results` reports every cell
> missing.

Selection and sharding flags:

| Flag | Effect |
|---|---|
| `--pipeline <name>` | restrict to one pipeline; repeatable |
| `--cell-id <id>` | restrict to exact catalog cell ids; repeatable |
| `--shard-count N` / `--shard-index I` | split the selected cells into `N` contiguous shards and run the zero-based shard `I` |
| `--dry-run` | list the selected cells and exit without training |
| `--force` | recompute even cached cells |

Sharding is how a 97-cell run is split across machines: launch the same command on
each host with the same `--shard-count` and a different `--shard-index`, then merge
the output directories. Because runs are keyed by `cell_id`, the shards cannot
collide.

### Tuning DataLoader workers

`SPIRAL_NUM_WORKERS` overrides the platform worker policy unconditionally. The
default picks a safe value per OS — `spawn` platforms and Python 3.14+ get fewer
workers, and pipelines that make large BLAS calls force single-process loading on
Linux to avoid an OpenBLAS fork deadlock. Set the variable only to force a specific
count or to reproduce that deadlock deliberately.

### Other subcommands

- `freeze-evidence` — refreeze the manifest, folds, environment and artifact
  checksums under `evidence/`. This rewrites provenance artifacts, so run it as a
  deliberate act when publishing a run, not as part of a routine reproduction.
- `dataset-fingerprint`, `verify-fingerprint` — see step 1 above.

### Which experiments need a GPU

| Pipeline | Training regime | GPU? |
|---|---|---|
| `dinov2_probe` | frozen public backbone → features → one logistic fit | **No** (CPU fine) |
| `folador_hog` | deterministic HOG features → SVM/RFC/KNN/CNN1D head | **No** |
| `dogan_triple` | three frozen public backbones + NCA + linear SVM | **CUDA required** — the TensorFlow NASNetMobile branch may still run on CPU |
| `huang_vgg19` | VGG19 end-to-end fine-tune, 50 epochs | **CUDA required** |

**The requirement is enforced, not advisory.** If a selected cell uses `huang_vgg19`
or `dogan_triple` and `torch.cuda.is_available()` is False, `reproduce` aborts with a
non-zero exit *before training starts* rather than falling back to CPU, where the
catalog takes days instead of hours. The message distinguishes the two causes — a
CPU-only torch wheel (`torch.version.cuda is None`) versus no visible device — because
they need different fixes. There is no override flag.

The two CPU pipelines are unaffected: `--pipeline dinov2_probe --pipeline folador_hog`
runs anywhere, and `--dry-run` lists the full catalog on any machine.

CPU-only reproduction (DINOv2 + Folador HOG) covers the paper's central H2/H4 story;
the GPU pipelines are needed for the literature in-domain ceilings and the full 3×3
matrices of Huang/Dogan.

### Huang VGG19 failure modes

`huang_vgg19` checks, **after** training finishes, that the run actually produced a
model. Two conditions abort the cell with a non-zero exit:

| Error | Meaning |
|---|---|
| `Huang VGG19 produced no usable gradient in any batch; the model never trained` | every batch's gradient was unusable, so no weight update ever landed |
| `Huang VGG19 collapsed to a single-class predictor on its own training data` | training completed, but the model predicts one class for its own training set |

**Both are guards, not crashes.** They exist because a collapsed fine-tune otherwise
produces plausible-looking but meaningless numbers. A single underflowed batch is
benign and expected near convergence under mixed precision — the check is at the
model level precisely so that normal underflow does not trip it.

If either fires, the cell has no valid result: re-run it rather than treating the
partial output as a finding. The run is idempotent, so re-running skips every cell
that already succeeded.

On Apple Silicon, `dinov2_probe` still uses the Metal Performance Shaders (`mps`)
backend automatically. `huang_vgg19` and `dogan_triple` do **not** — the CUDA
requirement above means they refuse to run on an MPS-only host, so earlier macOS
reproductions of those two pipelines are no longer possible through this CLI.

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

`dogan_triple` is sensitive to the machine in one place. Its NCA step stops at a
different iteration when its matrix products are split across a different number of
threads, which moves every downstream probability. The pipeline therefore runs NCA on a
fixed 8 threads (`NCA_THREADS`), so a result does not depend on how many cores the
machine has. It can still change with the CPU model or the BLAS build, so reproducing a
published Dogan cell exactly needs the same CPU model the run recorded.
