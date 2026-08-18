# Cross-Dataset Robustness of Spiral Classifiers for Parkinson's Disease

Reproduction code for the paper *"Cross-Dataset Robustness of Spiral Classifiers
for Parkinson's Disease Diagnosis"* (NIATS-UFU).

The paper asks a simple question: **do spiral-drawing Parkinson's classifiers that
look good in-domain actually transfer to other datasets?** It evaluates four
pipelines across three datasets in a full train × test factorial:

- **Pipelines:** DINOv2 frozen linear probe (the proposed method), Huang VGG19,
  Dogan triple-CNN + NCA + SVM, Folador HOG + SVM.
- **Datasets:** Zham (public), Folador (public), Moura (NIATS-UFU clinical, on request).

> **Dataset names.** The paper calls the three datasets *Zham*, *Folador* and *Moura*; the code keys
> them `kaggle`, `folador` and `hemato`. The keys are frozen — they are hashed into `cell_id`, into
> every `image_id` and `subject_id`, and into the released artifact names — so they appear verbatim in
> every command, path and result file below. Read `hemato` as **Moura** throughout.
- **Findings:** strong in-domain ceilings (**H1**) collapse off-diagonal — spiral
  classifiers **do not transfer across datasets** (**H2**) — while the frozen
  DINOv2 probe matches fine-tuned CNNs at a fraction of the hyperparameter and
  compute cost (**H4**).

## What's in here

- `src/parkinson_spiral/` — the installable package: experiment protocol, the four
  pipelines, augmentation recipes, metrics, and the cross-dataset analysis.
- `results/` — local generated outputs: manifests, cell index files, and
  predictions from re-running experiments. Result parquet files are not shipped.
- `scripts/` — Layer-1 regeneration and Layer-2 re-training entry points.
- `notebooks/reproduce_experiments.ipynb` — the one notebook that runs everything,
  in the paper's naming.
- `models/` — tooling to export and use the released DINOv2 + Moura probe.
- `requirements-lock.txt` — fully pinned transitive dependency lock for exact
  Python 3.13 reproduction environments.
- `docs/` — [`REPRODUCE.md`](docs/REPRODUCE.md), [`DATA.md`](docs/DATA.md),
  [`GCP_COLLABORATOR.md`](docs/GCP_COLLABORATOR.md),
  [`PROVENANCE.md`](docs/PROVENANCE.md), and review
  [`DECISIONS.md`](docs/DECISIONS.md).

## Project structure

The reproduction package intentionally mirrors the main repository layout:

```text
paper_reproduction/
├── src/parkinson_spiral/
│   ├── protocol.py        # CellSpec + Pipeline contracts
│   ├── catalog.py         # declarative experiment catalog
│   ├── runner.py          # split resolution + idempotent cell execution
│   ├── data/              # dataset-specific manifest builders
│   ├── pipelines/         # one module per model family
│   └── recipes/           # one module per augmentation strategy
├── tests/                 # unit and integration checks
├── scripts/               # command-line reproduction entry points
├── docs/                  # data, provenance, and reproduction notes
└── models/                # optional released-model export/inference tooling
```

To add an experiment, define a `CellSpec` in `catalog.py`. To add a model, create a
module under `pipelines/` and register it in `pipelines/__init__.py`. To add an
augmentation strategy, create a module under `recipes/` and register it in
`recipes/__init__.py`.

## Quickstart — inspect the experiment catalog (no data, no GPU)

```bash
pip install -e .
python scripts/paper.py reproduce --dry-run
```

The dry run lists the paper's 97 experiment cells without requiring datasets,
training, or a GPU. Tables and figures can be regenerated after producing a local
`results/` directory with `cells.parquet` and `predictions/`.

## Re-train from scratch (optional)

Needs the datasets on disk (see [`docs/DATA.md`](docs/DATA.md)); some pipelines
need a GPU (see [`docs/REPRODUCE.md`](docs/REPRODUCE.md)).

```bash
pip install -e ".[train]"
python scripts/paper.py build-manifest --data-root <path> --out results/manifest.parquet
python scripts/paper.py reproduce --manifest results/manifest.parquet --out results
python scripts/paper.py verify-catalog --results results
python scripts/paper.py regenerate --results results
```

Re-training is idempotent (keyed by `cell_id`): interrupted runs resume, and
existing local predictions are skipped unless `--force` is passed.

## The released model

One trained artifact accompanies the paper: a logistic **linear probe** on a
**frozen, public DINOv2 ViT-B/14** backbone, fit on 100% of Moura. The backbone is
never shipped (downloaded from `torch.hub` at use time); only the ~KB sklearn probe
is ours. See [`models/MODEL_CARD.md`](models/MODEL_CARD.md).

```bash
pip install -e ".[train,hf]"
python models/predict.py --model-dir models/exported --images path/to/spirals/
```

> It is a **reference / exploration checkpoint, not a screening or diagnostic
> tool** — consistent with the paper's central finding that these classifiers do
> not transfer across datasets.

## Data

No images are distributed. Zham and Folador are public; Moura is IRB-restricted
and available on request from NIATS-UFU. See [`docs/DATA.md`](docs/DATA.md).

## Citation

If you use this code, please cite *"Cross-Dataset Robustness of Spiral Classifiers
for Parkinson's Disease Diagnosis"* (NIATS-UFU). See the paper for the full author
list and BibTeX.

## License

- **Code:** [`LICENSE`](LICENSE) — MIT.
- **Released model:** [`models/MODEL_LICENSE.md`](models/MODEL_LICENSE.md) —
  **OpenRAIL-M** with an added no-clinical-use restriction, reflecting the clinical
  provenance of its training data. See [`models/MODEL_CARD.md`](models/MODEL_CARD.md).
