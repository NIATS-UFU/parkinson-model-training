# Parkinson Model Train

[![on-merge](https://github.com/NIATS-UFU/parkinson-model-train/actions/workflows/on_merge.yaml/badge.svg)](https://github.com/NIATS-UFU/parkinson-model-train/actions/workflows/on_merge.yaml)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)

Parkinson model train

## Installation

```sh
uv pip install parkinson_model_train
```

## ⚡ Features

- [`pre-commit`](https://github.com/pre-commit/pre-commit) validates your commits.
- [`ruff`](https://github.com/charliermarsh/ruff) provides static code analysis and a code and import formatting using [`black`](https://github.com/psf/black) and [`isort`](https://github.com/PyCQA/isort) styles.
- [`pytest`](https://github.com/pytest-dev/pytest) runs your test suite.
- [`commitizen`](https://github.com/commitizen-tools/commitizen) ensures you follow [Conventional commit style](https://www.conventionalcommits.org/en/) and generates your release changelogs according to [SemVer](https://semver.org/)
- [`python-semantic-release`](https://python-semantic-release.readthedocs.io/en/latest/) CI/CD automatically increases version and create git releases based on commit messages following conventional commit style.
- Docker support

### Automated Versioning Strategy

- PR requests receive a tag based on <current_version>.dev0.<commit_hash>. This is not a proper tag, but can be used for temporary deployment in the future
- On merge:
   - Bump semantic version based on conventional commits
   - OR - patch release on merge if commits contain no hint to conventional commit.


## Recommended Tools

- Managing Python versions? [pyenv](https://github.com/pyenv/pyenv)
- Building a CLI? [typer](https://typer.tiangolo.com/)
- Logging? [Loguru](https://github.com/Delgan/loguru)

