# Parkinson Model Training

[![on-merge](https://github.com/NIATS-UFU/parkinson-model-training/actions/workflows/on_merge.yaml/badge.svg)](https://github.com/NIATS-UFU/parkinson-model-training/actions/workflows/on_merge.yaml)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Python](https://img.shields.io/badge/python-3.13-blue.svg)

Code for the paper *Cross-Dataset Robustness of Spiral Classifiers for Parkinson's Disease
Screening* (NIATS-UFU). The paper evaluates four spiral-drawing classifiers on three datasets,
training on each dataset and testing on each.

All of it lives in [`paper_reproduction/`](paper_reproduction/README.md): a self-contained Python
package (`parkinson_spiral`) with its own environment, lockfile, tests and licence.

| To | Read |
|---|---|
| See what the paper asks and what the code contains | [`paper_reproduction/README.md`](paper_reproduction/README.md) |
| Run the experiments | [`paper_reproduction/docs/REPRODUCE.md`](paper_reproduction/docs/REPRODUCE.md) |
| Get the datasets | [`paper_reproduction/docs/DATA.md`](paper_reproduction/docs/DATA.md) |
| See how each reported number is derived | [`paper_reproduction/docs/PROVENANCE.md`](paper_reproduction/docs/PROVENANCE.md) |
| Understand the design choices | [`paper_reproduction/docs/DECISIONS.md`](paper_reproduction/docs/DECISIONS.md) |

## Dataset names

The manuscript and this codebase name the same datasets differently. Documentation uses the paper's
name; commands and paths use the key verbatim.

| Key in code | Name in the paper |
|---|---|
| `kaggle` | Zham |
| `folador` | Folador |
| `hemato` | **Moura**, the NIATS-UFU clinical cohort, released as `moura-neto-2023-2025` |

`hemato` is a frozen identifier baked into published result files and the released model artifacts;
it is never renamed. Read the cohort's name as *Moura* wherever the token appears.

## Repository layout

```text
paper_reproduction/    # the paper's package, scripts, docs, tests and frozen evidence
Makefile               # lint, format, pre-commit and test targets
pyproject.toml         # tooling only: ruff and pre-commit, with their lockfile uv.lock
.github/workflows/     # CI on pull requests, release tag on merge
```

The `pyproject.toml` and `uv.lock` at the repository root install the development tools and nothing
else. The paper's dependencies are pinned in `paper_reproduction/pyproject.toml` and
`paper_reproduction/uv.lock`.

## Development

```sh
make install    # development tools
make lint       # ruff check
make test       # the paper_reproduction test suite, in its own environment
```

`make fmt` and `make check` rewrite Python files in place. Every file under
`paper_reproduction/src/parkinson_spiral/` is hashed into the signature recorded with each result,
so run them only when you mean to change that package.

## Upgrading from a 0.x release

Releases before 1.0.0 also shipped a general training package at the repository root: the Python
package `parkinson_model_train`, the `main-cli` command, `train_baseline.py` and a Dockerfile.
Release 1.0.0 removes them, so the repository holds the paper's code only. Nothing under
`paper_reproduction/` changed with the removal.

To keep using the training package, stay on a 0.x release, for example:

```sh
git checkout v0.5.0
```

## License

- **Code:** [`LICENSE`](LICENSE), MIT.
- **Trained models:** CC-BY-NC-4.0 (attribution, non-commercial use only), stated on their
  [Zenodo record](https://doi.org/10.5281/zenodo.23748799).
