# Cross-Dataset Robustness of Spiral Classifiers for Parkinson's Disease

Reproduction code for the paper *"Cross-Dataset Robustness of Spiral Classifiers
for Parkinson's Disease Diagnosis"* (NIATS-UFU).

The paper asks a simple question: **do spiral-drawing Parkinson's classifiers that
look good in-domain actually transfer to other datasets?** It evaluates four
pipelines across three datasets in a full train × test factorial:

- **Pipelines:** DINOv2 frozen linear probe (the proposed method), Huang VGG19,
  Dogan triple-CNN + NCA + SVM, Folador HOG + SVM.
- **Datasets:** Kaggle (public), Folador (public), Hemato (NIATS-UFU clinical, on request).
- **Findings:** strong in-domain ceilings (**H1**) collapse off-diagonal — spiral
  classifiers **do not transfer across datasets** (**H2**) — while the frozen
  DINOv2 probe matches fine-tuned CNNs at a fraction of the hyperparameter and
  compute cost (**H4**).

## What's in here

- `src/parkinson_spiral/` — the installable package: experiment protocol, the four
  pipelines, augmentation recipes, metrics, and the cross-dataset analysis.
- `results/` — the curated **97-cell** result set (predictions + cell index) that
  backs every table and figure. **No images** are shipped.
- `scripts/` — Layer-1 regeneration and Layer-2 re-training entry points.
- `notebooks/reproduce_experiments.ipynb` — the one notebook that runs everything,
  in the paper's naming.
- `models/` — tooling to export and use the released DINOv2 + Hemato probe.
- `docs/` — [`REPRODUCE.md`](docs/REPRODUCE.md), [`DATA.md`](docs/DATA.md),
  [`PROVENANCE.md`](docs/PROVENANCE.md).

## Quickstart — regenerate the paper's tables and figures (no data, no GPU)

```bash
pip install -e .
python scripts/regenerate.py        # -> paper_artifacts/  (seconds)
```

This is the headline reproducibility claim: every table and figure is rebuilt from
the shipped predictions, with no datasets, no GPU, and no training. Open
`notebooks/reproduce_experiments.ipynb` to do the same interactively and see each
table inline.

## Re-train from scratch (optional)

Needs the datasets on disk (see [`docs/DATA.md`](docs/DATA.md)); some pipelines
need a GPU (see [`docs/REPRODUCE.md`](docs/REPRODUCE.md)).

```bash
pip install -e ".[train]"
python scripts/build_manifest.py --data-root <path> --out results/manifest.parquet
python scripts/reproduce.py --manifest results/manifest.parquet --out runs/fresh
python scripts/verify_catalog.py
```

Re-training is idempotent (keyed by `cell_id`): interrupted runs resume, and
pointing it at the shipped `results/` is a no-op.

## The released model

One trained artifact accompanies the paper: a logistic **linear probe** on a
**frozen, public DINOv2 ViT-B/14** backbone, fit on 100% of Hemato. The backbone is
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

No images are distributed. Kaggle and Folador are public; Hemato is IRB-restricted
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
