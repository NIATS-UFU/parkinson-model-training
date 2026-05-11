# Set the shell for Makefile
SHELL := /bin/bash

# Variables
REPO_NAME := parkinson_model_train
PYTHON_SHORT_VERSION := $(shell echo 3.12 | grep -o '[0-9].[0-9]*')
PYTHON_PACKAGE_PATH := .venv/lib/python$(PYTHON_SHORT_VERSION)/site-packages
LOCKFILE := uv.lock
VIRTUAL_ENV=.venv
IMAGE_TAG ?= latest

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

## Remove virtual environment and lockfile
.PHONY: clean
clean:
	@echo "Cleaning lockfile and temporary files..."
	rm -rf .venv $(LOCKFILE)
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
test: install pytest

## Run pytest with coverage report
.PHONY: pytest
pytest:
	@echo "Running tests with coverage..."
	uv run --all-groups pytest --cov=$(REPO_NAME) --cov-config=pyproject.toml --cov-report=term-missing tests
	@echo "✅ Tests completed!"

## Run pytest with verbose output
.PHONY: pytest-loud
pytest-loud:
	@echo "Running tests in verbose mode..."
	uv run --all-groups pytest -s --cov=$(REPO_NAME) --cov-report=term-missing tests
	@echo "✅ Verbose tests completed!"

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

###########################
# Packaging and Docker    #
###########################

## Build a wheel package
.PHONY: build
build: clean-build
	@echo "Creating wheel file..."
	uv build
	@echo "✅ Build complete!"

## Clean up build artifacts
.PHONY: clean-build
clean-build:
	@echo "Removing build artifacts..."
	rm -rf dist
	@echo "✅ Build artifacts removed!"


###########################
# Docker Management       #
###########################

## Build Docker image
.PHONY: docker-build
docker-build: build
	@echo "Building Docker image..."
	docker build -t $(REPO_NAME):$(IMAGE_TAG) -f docker/Dockerfile .
	@echo "✅ Docker image built!"

## Run Docker container
.PHONY: docker-run
docker-run: docker-build
	@echo "Running Docker container..."
	docker run -d --name $(REPO_NAME) $(REPO_NAME)
	@echo "✅ Docker container is running!"

## Stop and remove Docker container
.PHONY: docker-stop
docker-stop:
	@echo "Stopping and removing Docker container..."
	docker stop $(REPO_NAME) || true
	docker rm $(REPO_NAME) || true
	@echo "✅ Docker container stopped and removed!"
