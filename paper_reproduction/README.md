# Cross-Dataset Robustness of Spiral Classifiers for Parkinson's Disease Screening

Reproduction code for the paper *"Cross-Dataset Robustness of Spiral Classifiers
for Parkinson's Disease Screening"* (NIATS-UFU).

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
- **Findings:** each pipeline is first reproduced in-domain (**H1**). Across datasets,
  **transfer depends on the target and on the pipeline** (**H2**): between the two
  public datasets every pipeline's accuracy is above the majority-class baseline, and
  when Moura is the test set no pipeline's is. The frozen
  DINOv2 probe matches fine-tuned CNNs at a fraction of the hyperparameter and
  compute cost (**H3**). The code and the `h4-cost` table name label this
  hypothesis H4; read it as the paper's H3.

## What's in here

- `src/parkinson_spiral/` — the installable package: experiment protocol, the four
  pipelines, augmentation recipes, metrics, and the cross-dataset analysis.
- `results/` — local generated outputs: manifests, cell index files, and
  predictions from re-running experiments. Result parquet files are not shipped.
- `scripts/` — Layer-1 regeneration and Layer-2 re-training entry points.
- `notebooks/reproduce_experiments.ipynb` — the one notebook that runs everything,
  in the paper's naming.
- `requirements-lock.txt` — fully pinned transitive dependency lock for exact
  Python 3.13 reproduction environments.
- `docs/` — [`REPRODUCE.md`](docs/REPRODUCE.md), [`DATA.md`](docs/DATA.md),
  [`PROVENANCE.md`](docs/PROVENANCE.md), and the design
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
└── docs/                  # data, provenance, and reproduction notes
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

## Trained models

The four models trained on all of Moura, one per pipeline, are published openly at
[doi:10.5281/zenodo.23748799](https://doi.org/10.5281/zenodo.23748799). Moura itself is available only on request, so these are the
models nobody can rebuild without it; the eight trained on Zham or Folador can be
rebuilt from the public data with `reproduce --save-models` (see
[`docs/REPRODUCE.md`](docs/REPRODUCE.md)). Unzip each model archive into one
`models/` folder, and the record's reference predictions into `reference/`. The
cells these models serve test on Zham and Folador, so checking them against the
paper's run needs only the public data:

```bash
uv sync --extra train --extra hf
python scripts/paper.py build-manifest --data-root <path with kaggle/ and folador/> --out results/manifest.parquet
python scripts/paper.py verify-models --models models --manifest results/manifest.parquet --results reference
python scripts/paper.py predict --model models/dinov2_probe-hemato --images path/to/spirals/ --out predictions.csv
```

`predict` writes the probability of Parkinson's for each image, preprocessed as in
training. The paper decides per participant: the mean over that participant's
drawings, thresholded at 0.5. `predict` does not know which drawings belong to whom,
so it makes no per-participant decision: score one folder per participant (or name the
files so `image_id`, the file name, shows the participant), then average and threshold.

> These are **reference checkpoints for research, not a screening or diagnostic
> tool**. The paper finds that when Moura, the clinical cohort, is the test set, no
> pipeline's accuracy is above the majority-class baseline.

## Data

No images are distributed. Zham and Folador are public; Moura is IRB-restricted
and available on request from NIATS-UFU. See [`docs/DATA.md`](docs/DATA.md).

## Citation

If you use this code, please cite:

> M. F. Nozella, F. Andreotti, M. De Vos and A. O. Andrade, "Cross-Dataset Robustness
> of Spiral Classifiers for Parkinson's Disease Screening" (submitted).

## License

- **Code:** [`LICENSE`](LICENSE) — MIT.
- **Trained models:** CC-BY-NC-4.0 (attribution, non-commercial use only), stated on their Zenodo record.
