load("@rules_python//python:pip.bzl", "pip_parse")

PIP_EXTRA_ARGS = [
    "--cache-dir=~/.cache/pip",
    "--extra-index-url=https://pypi.tuna.tsinghua.edu.cn/simple",
    "--verbose",
]

def pip_deps():
    pip_parse(
        name = "pip_cpu_torch",
        requirements_lock = "//open_source/deps:requirements_lock_torch_cpu.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 36000,
    )

    pip_parse(
        name = "pip_arm_torch",
        requirements_lock = "//open_source/deps:requirements_lock_torch_arm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 36000,
    )

    pip_parse(
        name = "pip_ppu_torch",
        requirements_lock = "//open_source/deps:requirements_lock_torch_gpu.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 36000,
    )

    pip_parse(
        name = "pip_gpu_cuda12_torch",
        requirements_lock = "//open_source/deps:requirements_lock_torch_gpu_cuda12.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 36000,
        quiet = False,
    )

    pip_parse(
        name = "pip_gpu_rocm_torch",
        requirements_lock = "//open_source/deps:requirements_lock_rocm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 12000,
    )
