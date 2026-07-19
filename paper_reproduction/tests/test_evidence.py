"""Evidence capture degrades gracefully when an optional probe tool is absent."""

import importlib.util
from pathlib import Path

_PAPER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "paper.py"


def _load_paper():
    spec = importlib.util.spec_from_file_location("paper_script", _PAPER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_command_output_returns_none_for_missing_binary():
    # A missing executable is a normal condition off its platform (e.g. the
    # macOS-only `system_profiler` on Linux/Windows). It must return None like a
    # non-zero exit, not raise FileNotFoundError and crash freeze-evidence.
    paper = _load_paper()
    assert paper._command_output("parkinson-no-such-binary-xyz123", "arg") is None


def test_hardware_profile_never_raises():
    # Off macOS the hardware profile degrades to an empty mapping; on macOS it is
    # populated. Either way it returns a dict rather than raising.
    paper = _load_paper()
    assert isinstance(paper._hardware_profile(), dict)
