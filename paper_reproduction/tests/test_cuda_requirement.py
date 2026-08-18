"""CUDA is a hard requirement for the GPU pipelines.

`huang_vgg19` and `dogan_triple` train on the GPU; on CPU they take days, so the
CLI aborts instead of falling back. `dinov2_probe` and `folador_hog` stay
CPU-runnable. `_cuda_requirement_error` is pure, so each branch is checked
directly rather than by manipulating the real host.
"""

from types import ModuleType

GPU = ["huang_vgg19"]


def test_cpu_pipelines_run_without_cuda(paper: ModuleType):
    # The documented CPU path (notebook section 1, DATA.md) must not regress.
    assert paper._cuda_requirement_error(gpu_pipelines=[], cuda_available=False, torch_cuda_version=None) is None


def test_gpu_pipeline_with_cuda_proceeds(paper: ModuleType):
    assert paper._cuda_requirement_error(gpu_pipelines=GPU, cuda_available=True, torch_cuda_version="12.9") is None


def test_cpu_only_torch_build_is_reported_as_a_reinstall(paper: ModuleType):
    message = paper._cuda_requirement_error(gpu_pipelines=GPU, cuda_available=False, torch_cuda_version=None)

    assert "huang_vgg19" in message
    assert "CPU-only build" in message
    assert "requirements-lock.txt" in message


def test_missing_device_is_reported_as_a_driver_problem(paper: ModuleType):
    message = paper._cuda_requirement_error(gpu_pipelines=GPU, cuda_available=False, torch_cuda_version="12.9")

    assert "12.9" in message
    assert "no device is visible" in message
    assert "nvidia-smi" in message


def test_every_failure_names_the_cpu_only_escape(paper: ModuleType):
    # A collaborator blocked here needs to know what they can still run.
    for cuda_version in (None, "12.9"):
        message = paper._cuda_requirement_error(
            gpu_pipelines=["dogan_triple", "huang_vgg19"],
            cuda_available=False,
            torch_cuda_version=cuda_version,
        )
        assert "--pipeline dinov2_probe --pipeline folador_hog" in message


def test_require_cuda_ignores_a_cpu_only_selection(paper: ModuleType):
    from parkinson_spiral.catalog import all_cells

    cpu_cells = [cell for cell in all_cells() if cell.pipeline_name in {"dinov2_probe", "folador_hog"}]

    # Returns without consulting torch at all, on any host.
    assert paper._require_cuda(cpu_cells) is None
