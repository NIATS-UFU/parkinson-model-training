# Reproducing the paper

*"Cross-Dataset Robustness of Spiral Classifiers for Parkinson's Disease Screening."*

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

### Statistics and dataset distances

```bash
python scripts/paper.py statistics --results results --out results/statistics
python scripts/paper.py distances --embeddings results/embeddings.parquet --out results/distances --figure
```

`statistics` (`src/parkinson_spiral/evaluation.py`) reads the same result directory
and writes, all at the participant level:

- `intervals.csv`: accuracy, sensitivity, specificity and ROC-AUC with a 95%
  percentile bootstrap over participants (1,000 resamples from the run's seed, one
  shared resample set per cell), for the 12 diagonals (five folds pooled) and the 24
  off-diagonal cells. `n_one_class_resamples` counts resamples holding one class;
  any count above zero makes that cell's sensitivity, specificity or AUC interval
  unreliable.
- `paired.csv`: each pipeline against DINOv2 on every off-diagonal cell, with an exact
  McNemar test, Holm's correction across the six comparisons of a test set, and the
  Quesenberry-Hurst interval for the difference in error rates, unadjusted and at the
  Holm level (Berrar and Lozano 2013).
- `baselines.csv`: each test set's majority-class accuracy.

`distances` (`src/parkinson_spiral/distances.py`) needs the frozen DINOv2 embedding
of every spiral, which `export-embeddings` writes. That step reads the images, so it
needs the datasets and the `train` extra:

```bash
python scripts/paper.py export-embeddings --manifest results/manifest.parquet --out results/embeddings.parquet
```

It writes `distances.csv`: the sliced Wasserstein-2 distance between each pair of
datasets, each participant being the mean embedding of their spirals. Each distance
is the square root of the mean squared distance over 200 draws of 14 participants
per dataset, with 1,000 projection directions. Each pair is set against two references: the floor, which is the same
measure between disjoint halves of one dataset, and a null, in which participants
are shuffled across datasets. The estimate does not settle at these sample sizes,
so read the distances as an ordering of the pairs, not as magnitudes. It also
writes `tsne.csv`, the 2-D t-SNE map of the spiral embeddings. With `--figure` it
draws that map to `fig_tsne.png`, which needs the `figures` extra
(`pip install -e ".[figures]"`).

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

To match the published cells exactly, run the full catalog in catalog order in one
process. The DINOv2 backbone is built once per process and then reused, and building
it draws from torch's random generator. The cells trained with augmentation draw from
that generator too, so an augmented DINOv2 cell that is the first DINOv2 cell of its
process (after a resume, under `--cell-id`, or at the start of a shard) can see a
different random stream than it did in the published run. This follows from how the
code is written; it has not been measured.

### Saving and verifying the trained models

`--save-models <dir>` saves the 12 models trained on all of one dataset, one per
pipeline and training dataset: the fits behind the off-diagonal cells. Each lands in
`<dir>/<pipeline>-<dataset>/` with a `model.json` card listing the cells it serves,
the SHA-256 of each file, the commit and the package versions. Saving needs the `hf`
extra (`skops`).

| Pipeline | Saved files | Not saved (public, downloaded at use) |
|---|---|---|
| `dinov2_probe` | `probe.skops` (logistic regression) | DINOv2 ViT-B/14 |
| `folador_hog` | `classifier.skops` (linear SVM, as its weight vector) | none |
| `dogan_triple` | `nca.skops`, `svm.skops` (linear SVM, as its weight vector) | DenseNet201, Xception, NASNetMobile |
| `huang_vgg19` | `vgg19_state_dict.pt` (the fine-tuned network) | none |

```bash
python scripts/paper.py reproduce --manifest results/manifest.parquet --out results --save-models models
python scripts/paper.py verify-models --models models --manifest results/manifest.parquet --results results
```

Run the full catalog in catalog order, as above, when the models must match the
published cells. A DINOv2 cell retrained on its own in a fresh process gave image
probabilities up to 0.023 away from the same cell inside a full catalog run; the
cause has not been isolated. A model is saved only
when its cell trains, so a cell served from cache saves nothing; use `--force` or an
empty `--out`. Every file is reloaded and compared with the state that was written
before its card is written.

A fitted linear SVM keeps its support vectors, the feature vectors of hundreds of
training drawings. The saved file keeps only the weight vector and intercept, in a
`LinearSVC` that computes the same decision, and the card lists those files under
`weight_vector_heads`. So no saved model holds a training drawing's features. Its
decision values match the fitted SVM to float rounding rather than bit for bit, and
`verify-models` reports that difference.

`verify-models` reloads each model, re-predicts the cells it serves, and compares
them with the run that trained it. It exits non-zero if any participant-level metric
differs, and reports the largest per-image probability difference alongside. Loading
checks each file against its card. skops loads only scikit-learn types it trusts,
and the network loads as tensors only, so a saved model cannot run code when it is
loaded.

`predict` scores new drawings with one saved model: every image directly inside a
folder (or one image), preprocessed as in training, written as `image_id,p_parkinson`.
The paper's participant decision is the mean over that participant's drawings,
thresholded at 0.5.

```bash
python scripts/paper.py predict --model models/dinov2_probe-hemato --images path/to/spirals/ --out predictions.csv
```

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
- `cost-profile` — measure each pipeline's training and inference cost, one fresh
  process per pipeline. Use it for costs rather than the cost columns of a full
  catalog run (`fit_seconds`, `infer_latency_ms_per_img`, the memory columns). In a
  full run the pipelines share one process: DINOv2 and Dogan reuse image features
  extracted by earlier cells, and memory peaks include what earlier cells allocated.
  `fit_was_cached` flags only a fit reused whole, so a row with it False can still
  hold a near-zero time.

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

CPU-only reproduction (DINOv2 + Folador HOG) covers the paper's central H2/H3 story;
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
requirement above means they refuse to run on an MPS-only host.

The notebook splits this for you: **Section 1** runs the CPU-only subset, and
**Section 2** runs the GPU pipelines behind a `torch.cuda.is_available()` guard.

---

## What "reproduces" means here

All reported numbers are **subject-level**, pooled across the cells that make up a
logical result (e.g. the 5 CV folds of a diagonal), with a percentile bootstrap
95% CI (`n_resamples=1000`, `seed=20260516`). Re-deriving them from generated
predictions is deterministic and exact once those predictions exist locally. How
each number is derived is described in [`PROVENANCE.md`](PROVENANCE.md).

`dogan_triple` is sensitive to the machine. Its NCA step stops at a different
iteration when its matrix products are split across a different number of threads,
which moves every downstream probability. The pipeline therefore runs NCA on a fixed 8
threads (`NCA_THREADS`). That removes most of the dependence on the core count, but not
all of it: on one CPU model with 4 and with 8 cores visible, most Dogan cells gave
identical probabilities and a few did not. Results can also change with the CPU model or
the BLAS build. Reproducing a published Dogan cell exactly therefore needs the machine
that produced it: the published numbers come from 8 vCPUs of an Intel Broadwell CPU (a
GCP `n1-standard-8` with one Tesla T4). The frozen evidence records that machine under
`hardware_profile` in `evidence/environment.json` (`CPU`, `CPU model number`, `Logical
processors`); compare your own with `lscpu` and `nproc`.

`NCA_THREADS` is a fixed count, not the machine's. During the NCA fit and prediction it
overrides `OPENBLAS_NUM_THREADS` and `OMP_NUM_THREADS`, and it applies to every thread
pool loaded in the process, OpenMP pools included, not only BLAS. On a host with fewer
than 8 cores those steps therefore run more threads than there are cores, which can
slow them. This is deliberate: the fixed count keeps the split of NCA's matrix products
the same as on the published machine.
