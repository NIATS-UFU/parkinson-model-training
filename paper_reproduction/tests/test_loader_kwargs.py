"""DataLoader worker selection.

Linux forks workers by default, but a BLAS-heavy pipeline opts out of forking
(`forbid_fork=True`) to avoid the OpenBLAS fork deadlock, and `SPIRAL_NUM_WORKERS`
overrides everything. `_resolve_loader_kwargs` is pure, so each platform branch is
checked directly rather than by manipulating the real host.
"""

from parkinson_spiral.pipelines import _resolve_loader_kwargs

LINUX = {"os_name": "posix", "system": "Linux", "python_version": (3, 13)}


def test_linux_forks_two_workers_by_default():
    kw = _resolve_loader_kwargs(**LINUX, forbid_fork=False, override=None)
    assert kw == {"num_workers": 2, "persistent_workers": True}


def test_linux_forbid_fork_forces_single_process():
    # The fix: a pipeline whose later stage makes a large BLAS call (Dogan's NCA)
    # must not fork a worker on Linux, or the poisoned OpenBLAS threadpool deadlocks.
    kw = _resolve_loader_kwargs(**LINUX, forbid_fork=True, override=None)
    assert kw == {"num_workers": 0}


def test_windows_uses_spawn_regardless_of_forbid_fork():
    for forbid in (False, True):
        kw = _resolve_loader_kwargs(
            os_name="nt", system="Windows", python_version=(3, 13), forbid_fork=forbid, override=None
        )
        assert kw == {"num_workers": 0}


def test_macos_uses_spawn():
    kw = _resolve_loader_kwargs(
        os_name="posix", system="Darwin", python_version=(3, 13), forbid_fork=False, override=None
    )
    assert kw == {"num_workers": 0}


def test_python_314_uses_spawn_on_linux():
    kw = _resolve_loader_kwargs(
        os_name="posix", system="Linux", python_version=(3, 14), forbid_fork=False, override=None
    )
    assert kw == {"num_workers": 0}


def test_env_override_zero_wins_over_default_fork():
    kw = _resolve_loader_kwargs(**LINUX, forbid_fork=False, override="0")
    assert kw == {"num_workers": 0}


def test_env_override_count_wins_even_over_forbid_fork():
    # An explicit worker count is a deliberate escape hatch and takes precedence.
    kw = _resolve_loader_kwargs(**LINUX, forbid_fork=True, override="4")
    assert kw == {"num_workers": 4, "persistent_workers": True}
