"""Evidence capture degrades gracefully when an optional probe tool is absent."""

from types import ModuleType


def test_command_output_returns_none_for_missing_binary(paper: ModuleType):
    # A missing executable is a normal condition off its platform (e.g. the
    # macOS-only `system_profiler` on Linux/Windows). It must return None like a
    # non-zero exit, not raise FileNotFoundError and crash freeze-evidence.
    assert paper._command_output("parkinson-no-such-binary-xyz123", "arg") is None


def test_hardware_profile_never_raises(paper: ModuleType):
    # Off macOS the hardware profile degrades to an empty mapping; on macOS it is
    # populated. Either way it returns a dict rather than raising.
    assert isinstance(paper._hardware_profile(), dict)
