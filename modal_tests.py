"""Infrastructure only: upload current sources and run the existing tests on Modal.

From assignment2-systems:
    uvx modal run modal_tests.py
Append --k 'test_flash_forward_pass_triton and False' to select one case,
or --collect-only to inspect which tests would run.

For an interactive shell in the same GPU environment:
    uvx modal shell modal_tests.py::run_tests
Then use /workspace/assignment2-systems/.venv/bin/python -m pytest (the
project packages are supplied through PYTHONPATH, rather than installed).

If this file lives outside the repository, set CS336_PROJECT_DIR to the
assignment2-systems directory and pass this file's absolute path to Modal.

Source files are attached at container startup, after dependency installation.
Every new `modal run` uploads the current files; a running invocation is a snapshot.
"""

import hashlib
import os
import shlex
import subprocess
from pathlib import Path

import modal

REMOTE_PROJECT = Path("/workspace/assignment2-systems")
REMOTE_BASICS = REMOTE_PROJECT.parent / "stanford-cs336-assignment1-basics"
DEFAULT_K = "test_flash_forward_pass_triton"
IGNORE = ["**/__pycache__/**", "**/*.pyc", "**/.DS_Store"]


def build_image():
    image = (
        modal.Image.debian_slim(python_version="3.13")
        .apt_install("gcc", "libgomp1")
        .uv_pip_install("uv==0.12.8")
    )
    if not modal.is_local():
        return image

    project = Path(os.environ.get("CS336_PROJECT_DIR", Path(__file__).resolve().parent)).resolve()
    basics = project.parent / "stanford-cs336-assignment1-basics"
    for local, remote in ((project, REMOTE_PROJECT), (basics, REMOTE_BASICS)):
        for name in ("pyproject.toml", "README.md"):
            source = local / name
            if not source.is_file():
                raise FileNotFoundError(f"Missing {source}. Set CS336_PROJECT_DIR to assignment2-systems.")
            image = image.add_local_file(source, str(remote / name), copy=True)

    # Cache third-party dependencies using metadata only. The local packages are
    # supplied via PYTHONPATH below, so changing their code cannot invalidate this layer.
    image = (
        image.add_local_file(project / "uv.lock", str(REMOTE_PROJECT / "uv.lock"), copy=True)
        .run_commands(
            f"cd {REMOTE_PROJECT} && uv sync --frozen --no-dev "
            "--no-install-project --no-install-package cs336-basics"
        )
        .env({
            "PYTHONUNBUFFERED": "1",
            "WANDB_MODE": "disabled",
            "PYTHONPATH": f"{REMOTE_PROJECT}:{REMOTE_BASICS}",
        })
    )
    for local, remote in (
        (project / "cs336_systems", REMOTE_PROJECT / "cs336_systems"),
        (project / "tests", REMOTE_PROJECT / "tests"),
        (basics / "cs336_basics", REMOTE_BASICS / "cs336_basics"),
    ):
        if not local.is_dir():
            raise FileNotFoundError(f"Missing source directory: {local}")
        image = image.add_local_dir(local, str(remote), copy=False, ignore=IGNORE)
    return image


image = build_image()
app = modal.App("cs336-gpu-tests")


@app.function(
    image=image,
    gpu="A100",
    cpu=4,
    memory=16384,
    timeout=600,
    max_containers=1,
    scaledown_window=2,
)
def run_tests(k: str = DEFAULT_K, test: str = "tests/test_attention.py", collect_only: bool = False) -> int:
    """Run pytest in a fresh subprocess and stream its unmodified output."""
    python = str(REMOTE_PROJECT / ".venv/bin/python")
    kernel = REMOTE_PROJECT / "cs336_systems/flash_attention2.py"
    print(f"Uploaded flash_attention2.py SHA256: {hashlib.sha256(kernel.read_bytes()).hexdigest()}", flush=True)

    # This runs remotely in the project environment. No torch/Triton imports on Mac.
    subprocess.run(
        [python, "-c", (
            "import torch, triton; "
            "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
            "print('GPU:', torch.cuda.get_device_name(0)); "
            "print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'Triton:', triton.__version__)"
        )],
        cwd=REMOTE_PROJECT,
        check=True,
    )
    command = [python, "-m", "pytest", test, "-vv"]
    if k:
        command.extend(["-k", k])
    if collect_only:
        command.append("--collect-only")
    print(f"Running: {shlex.join(command)}", flush=True)
    return subprocess.run(command, cwd=REMOTE_PROJECT, check=False).returncode


@app.local_entrypoint()
def main(k: str = DEFAULT_K, test: str = "tests/test_attention.py", collect_only: bool = False):
    exit_code = run_tests.remote(k, test, collect_only)
    print(f"pytest exit code: {exit_code}", flush=True)
    if exit_code:
        raise SystemExit(exit_code)
