# Set the shell for Makefile
SHELL := /bin/bash

# Variables
PYTHON_SHORT_VERSION := $(shell echo 3.13 | grep -o '[0-9].[0-9]*')
LOCKFILE := uv.lock
VIRTUAL_ENV=.venv

###########################
# Environment Setup       #
###########################

# Default target
.PHONY: all
all: install test
	@echo "✅ All tasks completed!"

## Install the virtual environment and dependencies
.PHONY: install
install: uv-sync
	@echo "✅ Environment setup complete!"


## Create Python virtual environment
$(VIRTUAL_ENV):
	@echo "Creating virtual environment using uv..."
	uv venv create --python $(PYTHON_SHORT_VERSION)

## Check if `uv` is installed, install it if missing
.PHONY: check-install-uv
check-install-uv:
	@echo "Checking for uv installation..."
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		source $HOME/.local/bin/env; \
		echo "✅ uv installed successfully"; \
	else \
		echo "✅ uv is already installed"; \
	fi

## Install dependencies using `uv`
.PHONY: uv-sync
uv-sync: check-install-uv $(VIRTUAL_ENV)
	@echo "Installing dependencies from $(LOCKFILE)..."
	uv sync --all-groups
	@echo "✅ Dependencies installed!"

## Remove the virtual environment while preserving the reproducibility lockfile
.PHONY: clean
clean:
	@echo "Cleaning virtual environment and temporary files..."
	rm -rf .venv
	@echo "✅ Cleanup complete!"

###########################
# Linting and Testing     #
###########################

## Run all pre-commit hooks
.PHONY: run-pre-commit
run-pre-commit: install
	@echo "Running pre-commit hooks on all files..."
	uv run --all-groups python -m pre_commit run --all-files
	@echo "✅ Pre-commit checks complete!"

## Run all tests
.PHONY: test
test: install paper-reproduction-test

## Run paper reproduction package tests in its own environment
.PHONY: paper-reproduction-test
paper-reproduction-test:
	@echo "Running paper reproduction tests..."
	uv run --project paper_reproduction --with pytest python -m pytest paper_reproduction/tests
	@echo "✅ Paper reproduction tests completed!"

## Run paper reproduction package tests with the extras the saved-model tests need
# Those tests skip themselves when skops or torch is missing, so the import line makes a missing one fail.
.PHONY: paper-reproduction-test-models
paper-reproduction-test-models:
	@echo "Running paper reproduction tests with the train and hf extras..."
	uv run --project paper_reproduction --extra train --extra hf python -c "import skops, torch"
	uv run --project paper_reproduction --extra train --extra hf --with pytest python -m pytest paper_reproduction/tests
	@echo "✅ Paper reproduction tests with extras completed!"

## Format code with ruff
.PHONY: fmt
fmt: install
	uv run --all-groups ruff format .
	uv run --all-groups ruff check --fix .
	@echo "✅ Formatting complete!"

## Lint code with ruff
.PHONY: lint
lint: install
	uv run --all-groups ruff check .
	@echo "✅ Lint complete!"

## Type check (add mypy to dev deps to enable)
.PHONY: type
type:
	@echo "⚠️  Type checking not configured. Add mypy to dev dependencies to enable."

## Run all checks: fmt, lint, type, test
.PHONY: check
check: fmt lint type test
	@echo "✅ All checks passed!"
