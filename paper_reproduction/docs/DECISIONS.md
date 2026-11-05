# Design decisions

The engineering decisions behind the paper reproduction package, and the protocol
rules they fix.

## Runtime

- Python is standardized on **3.13** (`>=3.13,<3.14`), for the paper reproduction
  package and for the repository's development tools.
- `from __future__ import annotations` imports are intentionally absent because
  Python 3.13 is the minimum runtime.
- The paper reproduction package caps Python at `<3.14` until the data-loading
  stack and worker-start behavior are validated under Python 3.14+.

## Reproducibility

- Layer-1 dependencies are pinned in `pyproject.toml` and `requirements.txt` so
  table and figure regeneration uses a stable numerical stack.
- `requirements-lock.txt` records the complete transitive environment for exact
  Python 3.13 reruns, including the Layer-2 training and model-saving extras.
- The DINOv2 `torch.hub` download is pinned to upstream commit
  `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; torchvision, timm, and TensorFlow/Keras
  pretrained loaders are controlled by the locked package versions.
- Layer-2 training extras are pinned to avoid silent changes in framework,
  pretrained-weight, and augmentation behavior.
- `scripts/paper.py verify-catalog` validates both the catalog cell IDs and the
  prediction parquet files for presence, schema, non-empty contents, and matching
  `cell_id` values.
- `scripts/paper.py reproduce` skips cells whose train/test datasets are absent from the
  manifest, making partial public-data reproductions explicit instead of failing
  midway through the catalog.

## Protocol rules

Three rules decide who counts as a participant and which fold they land in. Every
result depends on them.

### One identity per Zham participant (`data/kaggle.py`)

A Zham file name carries the participant, the class and the task: `V01HE01.png` is
participant `V01`, healthy, spiral (`E`), and `V01HO01.png` is the same participant's
wave (`O`). The loader reads both tasks with one pattern:

```python
_FNAME_RE = re.compile(r"^(V\d+)([PH])[EO]\d+\.(?:png|jpe?g)$", re.IGNORECASE)
```

Every image of a person therefore shares one `subject_id` (`kaggle:V01H`), and the
dataset has 28 participants (13 healthy, 15 Parkinson's). This matters even for
spiral-only experiments: folds are assigned on the full manifest, before a cell's task
filter is applied, so a participant's fold must not depend on how the wave files are
named.

### Folds are assigned per participant, not per image (`data/splits.py`)

Fold assignment works on a table with one row per participant, and the fold is then
mapped back to all of that participant's images:

```python
rows = out.loc[mask, ["subject_id", "class_label"]].drop_duplicates()
subjects = rows.drop_duplicates("subject_id").reset_index(drop=True)
y = (subjects["class_label"] == "parkinson").astype(int).to_numpy()
splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
```

Each participant counts once, however many images they have, so the five folds are
class-stratified at the participant level. That is the level every reported metric is
computed at. Splitting image rows instead would let a participant with many drawings
weigh more in the split than one with a single drawing.

### Every Moura manifest row is a file on disk (`data/hemato.py`)

In the layout that ships `all_metadata.parquet`, a row whose recorded path does not
resolve to a real file is dropped:

```python
if not abs_path.is_file():
    continue
```

The manifest therefore never points a pipeline at a missing image.

> Row counts depend on the Moura layout and crop you hold, and `session_id` does not
> mean the same thing across layouts. Fingerprint your own copy with
> `scripts/paper.py dataset-fingerprint` rather than comparing against a number quoted
> elsewhere. See [`DATA.md`](DATA.md).

## Experiment semantics

- Cross-dataset cells with the same pipeline, recipe, seed, and training source
  reuse one fitted source model for both foreign test datasets. This preserves
  the intended matrix semantics: a matrix row is one trained source model
  evaluated against multiple targets, not two independently randomized fits.
- The Dogan paper reports augmentation before splitting, but does not publish the
  random split or variant assignment needed to reproduce that leakage. The
  released fidelity cell therefore uses the public released split and applies
  augmentation only to training images. The paper's 99.39% is retained only as
  an unverified, non-participant-disjoint upper bound.
- All three Dogan backbones are frozen public feature extractors in the paper
  catalog. TensorFlow 2.21 exposes no Apple Metal device, so literal NASNetMobile
  fine-tuning would leave one branch on CPU, and the three backbones would not be
  trained under one regime.
- The catalog is the source of truth for the 97 full-paper cells. Partial runs
  are allowed only as local reruns when the manifest lacks restricted data.

## Tests and CI

- Root `make test` runs the `paper_reproduction/tests` suite in the reproduction
  package environment.
- Reproduction tests may skip result-dependent assertions when local
  generated parquet outputs are absent.

## Code structure

- Small helpers are kept where they encode domain concepts, such as catalog group
  construction and metric aggregation.
- The paper reproduction entry points are consolidated in `scripts/paper.py`, so one
  logical CLI is not split across several small scripts.
