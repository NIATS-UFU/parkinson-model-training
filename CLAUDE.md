# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**parkinson-model-training** — training and paper-reproduction code for NIATS-UFU spiral-drawing
Parkinson's disease classification.

- **Author:** Fernando Andreotti <andreotti@ufu.br>
- **Python:** 3.13 (`requires-python = ">=3.13,<3.14"`)
- **Repo:** https://github.com/NIATS-UFU/parkinson-model-training

The repository holds two independent halves:

- **`src/parkinson_model_train/`** — the training baseline exposed through the `main-cli` command,
  with its own `pyproject.toml` and `uv.lock` at the repository root.
- **`paper_reproduction/`** — a self-contained package (`parkinson_spiral`) that reproduces the
  published cross-dataset benchmark. It has its own `pyproject.toml`, lockfile and test suite, and is
  installed into a separate environment.

## Commands

```bash
make install               # Install deps with uv (always run first)
make fmt                   # Format with ruff (auto-fixes in place — see the warning below)
make lint                  # Lint with ruff
make type                  # Type-check
make test                  # Run BOTH suites: root pytest + paper_reproduction
make pytest                # Root suite only, with coverage
make paper-reproduction-test  # paper_reproduction suite only, in its own env
make paper-reproduction-test-models  # same suite with the train and hf extras, so the saved-model tests run
make pytest-loud           # Root suite, verbose, no capture
make check                 # fmt + lint + type + test — use before presenting work
make run-pre-commit        # Run all pre-commit hooks over every file
make build                 # Build wheel to dist/
make docker-build          # Build Docker image (requires make build first)
make docker-run            # Build and run the image
make docker-stop           # Stop the running container
make clean                 # Remove the virtual environment, preserving uv.lock
```

Run a single test:

```bash
uv run pytest tests/test_train.py::test_name -v
```

The two suites live in different environments. `uv run pytest` from the repository root sees only
`tests/` (`testpaths = ["tests"]`); the reproduction suite needs
`uv run --project paper_reproduction --with pytest python -m pytest paper_reproduction/tests`.

> **`make fmt` and `make check` rewrite `.py` files in place across the whole tree.** Every file under
> `paper_reproduction/src/parkinson_spiral/` is byte-hashed into the published run signature (see
> *Reproducibility invariants* below), so a reformat there silently invalidates it. Prefer `make lint`
> and `make test` when you are not deliberately changing that package.

## Architecture

```
src/parkinson_model_train/     # Root training package — CLI entry point `main-cli`
  main.py                      # Typer CLI shell: argument parsing, dispatch
  train.py                     # Training loop, checkpointing, run summary
  evaluation.py                # Metric computation
  reporting.py                 # SVG plot output
  data/
    data_models.py             # ImageRecord and friends
    data_utils.py              # Manifest building, subject-level splitting
    datasets/                  # Per-layout loaders: hemato, niats_folador, utils
train_baseline.py              # Standalone cross-validation baseline script (not part of the CLI)
tests/                         # Root suite: conftest, test_baseline, test_data,
                               #   test_evaluation, test_train
docs/training_walkthrough.md   # Notebook-style training guide
notebooks/colab_training.ipynb # Colab GPU training notebook
data/README.md                 # Expected on-disk dataset layouts

paper_reproduction/            # Self-contained benchmark reproduction package
  src/parkinson_spiral/        # protocol, catalog, runner, analysis, metrics, compute, evaluation, distances,
                               #   data/ (kaggle, folador, hemato, unified, splits, dataset),
                               #   pipelines/ (dinov2_probe, huang_vgg19, dogan_triple, folador_hog),
                               #   recipes/ (aug_none, aug_huang, aug_dogan)
  scripts/paper.py             # Single CLI for every reproduction subcommand
  evidence/                    # Frozen provenance artifacts — see the warning below
  docs/                        # DATA, REPRODUCE, DECISIONS, PROVENANCE
  tests/                       # Reproduction suite

docker/Dockerfile              # Installs pre-built wheel, entrypoint: main-cli
.github/workflows/
  on_pr.yaml                   # CI on PR: pre-commit + pytest + dev version tag
  on_merge.yaml                # CI on merge: semantic-release auto-version bump
```

**Versioning:** Automated via `python-semantic-release` + `hatch-vcs`. Version is derived from git tags — never edit manually. Use conventional commits to control bumps: `feat:` → minor, `fix:` → patch, `feat!:` → major. A patch bump is forced on every merge if no conventional hint is found.

**CLI entry point:** `main-cli` (defined in `pyproject.toml` `[project.scripts]`). Add sub-commands with `typer`.

## Dataset naming

The manuscript and this codebase use different names for the same three datasets. In prose, use the
paper's name; in code, paths and commands, use the key verbatim.

| Key in code | Name in the paper | Notes |
|---|---|---|
| `kaggle` | Zham | Public Kaggle "Parkinson's Drawings" set |
| `folador` | Folador | Public HOG-IA handwriting dataset |
| `hemato` | **Moura** | NIATS-UFU clinical cohort, released as `moura-neto-2023-2025`; IRB-restricted |

**`hemato` is a frozen identifier and must not be renamed.** It is not merely a label: it is hashed
into `cell_id` (`protocol.py`) for 46 of the 97 benchmark cells, into every `image_id` and
`subject_id` (`data/unified.py`), and it gates the `VALID_DATASETS` / `VALID_SOURCES` validators. It
is also the on-disk directory name, the `main-cli create-manifest` argument, and part of the
released model names (`<pipeline>-hemato`).
Renaming it would invalidate the published results and require re-running the full benchmark on
IRB-restricted data. Refer to the cohort as *Moura* in documentation; leave the token alone.

## Reproducibility invariants

`paper_reproduction` records a signature with every result cell so a run can be traced back to the
exact code that produced it. Two consequences matter when editing:

- **`runner._source_sha256()` hashes the relative path and raw bytes of every `.py` file under
  `paper_reproduction/src/parkinson_spiral/`.** Changing a comment, a docstring, or trailing
  whitespace there — or renaming a module — changes the signature of all 97 cells. Treat that package
  as immutable unless you intend to supersede a published run.
- **`paper_reproduction/evidence/` holds frozen artifacts** whose SHA-256 digests are recorded in
  `evidence/checksums.sha256`. Never hand-edit them; regenerate them with
  `scripts/paper.py freeze-evidence` as a deliberate act.

## Code Rules

- **Python 3.13.** Prefer standard library; only add a dependency when the benefit is clear and explicit.
- **Functions over classes.** No service/manager/handler classes unless you need identity + polymorphism. Use `@dataclass(frozen=True)` or `NamedTuple` for structured data.
- **Fail fast.** Raise clear, specific exceptions on invalid input. No silent fallbacks, no broad `try/except`. Catch only when you can add meaningful context or convert to a domain error.
- **Go straight to the expected path.** Implement the primary workflow cleanly. No speculative alternative behaviours unless explicitly required.
- **Minimal surface area.** One change-set does one thing. Touch the fewest files needed.
- **Type hints** on all public functions.

## Testing Rules

- **pytest only.** No `unittest`, no `unittest.mock`, no patching-based approaches.
- Use real code paths and real objects. Inject dependencies via function parameters instead of mocking.
- Structure tests: Arrange / Act / Assert. Assert on outcomes, not implementation details.
- Unsupported edge cases should raise — tests must assert the exception type and message.

## Documentation Rules

- Docstrings required on public functions and modules. Keep them short — what it does and how to use it.
- Comments explain **why**, not what. Full sentences, used sparingly.
- Do not update docs unless public behaviour changed.

## Before presenting any change

1. Run `make check` — confirm it passes. If your change must not touch `parkinson_spiral`, run `make lint && make test` instead.
2. State the assumption if something is ambiguous and proceed; don't ask for confirmation on small details.
3. If adding a dependency, explain why explicitly.

## Ask before doing

- Adding or removing dependencies
- Changing the public API of existing functions
- Renaming modules or restructuring `src/`
- Modifying CI workflows
- Bumping versions manually
- Editing anything under `paper_reproduction/src/parkinson_spiral/` or `paper_reproduction/evidence/`
