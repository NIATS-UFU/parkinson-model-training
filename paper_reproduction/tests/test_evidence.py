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


def test_linux_hardware_profile_reads_cpu_memory_and_count(paper: ModuleType):
    cpuinfo = (
        "processor\t: 0\nmodel\t\t: 79\nmodel name\t: Intel(R) Xeon(R) CPU @ 2.20GHz\n\n"
        "processor\t: 1\nmodel\t\t: 79\nmodel name\t: Intel(R) Xeon(R) CPU @ 2.20GHz\n"
    )
    meminfo = "MemTotal:       30812345 kB\nMemFree:        1000 kB\n"

    profile = paper._linux_hardware_profile(cpuinfo, meminfo)

    assert profile == {
        "CPU": "Intel(R) Xeon(R) CPU @ 2.20GHz",
        "CPU model number": "79",
        "Logical processors": "2",
        "Memory": "30812345 kB",
    }


def test_run_device_policy_follows_the_selected_device(paper: ModuleType):
    cuda = paper._run_device_policy("cuda", "[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]")
    mps = paper._run_device_policy("mps", None)

    assert cuda == {
        "dinov2_probe": "PyTorch CUDA",
        "huang_vgg19": "PyTorch CUDA, mixed precision",
        "dogan_triple_torch": "PyTorch CUDA",
        "dogan_triple_nasnetmobile": "TensorFlow CPU",
        "folador_hog": "CPU",
    }
    assert mps["huang_vgg19"] == "PyTorch MPS, full precision"


def test_run_device_policy_reports_a_tensorflow_gpu(paper: ModuleType):
    devices = "[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]"

    assert paper._run_device_policy("cuda", devices)["dogan_triple_nasnetmobile"] == "TensorFlow GPU"
