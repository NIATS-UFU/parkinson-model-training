# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project

**Parkinson Model Train** — Parkinson model train

- **Author:** Fernando Andreotti <andreotti@ufu.br>
- **Python:** 3.12+
- **Repo:** https://github.com/NIATS-UFU/parkinson-model-train

## Commands

```bash
make install        # Install deps with uv (always run first)
make fmt            # Format with ruff (auto-fixes in place)
make lint           # Lint with ruff
make test           # Run pytest with coverage
make check          # Run fmt + lint + type + test — use before presenting work
make run-pre-commit # Run all pre-commit hooks
make build          # Build wheel to dist/
make docker-build   # Build Docker image (requires make build first)
make clean          # Remove venv and lockfile
```

Run a single test:

```bash
uv run pytest tests/test_main.py::test_name -v
```

## Architecture

```
src/parkinson_model_train/
  __init__.py     # Package init (version exposed here)
  main.py         # Typer CLI entry point + core logic
tests/
  conftest.py     # Shared pytest fixtures
  test_main.py    # Tests for main.py
docker/
  Dockerfile      # Installs pre-built wheel, entrypoint: main-cli
.github/workflows/
  on_pr.yaml      # CI on PR: pre-commit + pytest + dev version tag
  on_merge.yaml   # CI on merge: semantic-release auto-version bump
```

**Versioning:** Automated via `python-semantic-release` + `hatch-vcs`. Version is derived from git tags — never edit manually. Use conventional commits to control bumps: `feat:` → minor, `fix:` → patch, `feat!:` → major. A patch bump is forced on every merge if no conventional hint is found.

**CLI entry point:** `main-cli` (defined in `pyproject.toml` `[project.scripts]`). Add sub-commands with `typer`.

## Code Rules

- **Python 3.12+.** Prefer standard library; only add a dependency when the benefit is clear and explicit.
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

## Working with Fernando Andreotti

### Before presenting any change

1. Run `make check` — confirm it passes.
2. State the assumption if something is ambiguous and proceed; don't ask for confirmation on small details.
3. If adding a dependency, explain why explicitly.

### Communication style

- Lead with the answer or the change, not the reasoning.
- Flag blockers immediately; don't explore silently.
- Be direct and concise.

### Ask before doing

- Adding or removing dependencies
- Changing the public API of existing functions
- Renaming modules or restructuring `src/`
- Modifying CI workflows
- Bumping versions manually
