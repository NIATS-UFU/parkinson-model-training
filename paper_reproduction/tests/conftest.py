"""Shared fixtures for paper_reproduction tests."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_PAPER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "paper.py"


@pytest.fixture
def paper() -> ModuleType:
    """Load `scripts/paper.py` as a module.

    `scripts/` isn't on `pythonpath` (only `src` is), so tests that need to reach
    into the script import it directly from its file path instead.
    """
    spec = importlib.util.spec_from_file_location("paper_script", _PAPER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
