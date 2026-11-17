"""Run the existing distributed benchmark on a single Modal machine."""

import json
import subprocess

import modal

from modal_common import REMOTE_PROJECT, image

app = modal.App("cs336-distributed")
WORLD_SIZES = (2, 4, 6)
DEFAULT_GPU = "A100-80GB"


def parse_sizes(value: str, flag: str) -> list[int]:
    try:
        sizes = [int(part) for part in value.split(",")]
    except ValueError as error:
        raise ValueError(f"{flag} must be a comma-separated list of integers") from error
    if any(size <= 0 for size in sizes) or len(sizes) != len(set(sizes)):
        raise ValueError(f"{flag} must contain distinct positive integers")
    return sizes


@app.function(image=image, timeout=900, scaledown_window=2, max_containers=1)
def run(world_size: int, data_sizes: list[int], expected_model: str | None) -> str:
    python = str(REMOTE_PROJECT / ".venv/bin/python")
    gpu_info = (
        "import json, torch; "
        "print(json.dumps([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))"
    )
    devices = json.loads(subprocess.check_output([python, "-c", gpu_info], cwd=REMOTE_PROJECT, text=True))
    print(f"CUDA devices: {devices}", flush=True)
    if len(devices) != world_size:
        raise RuntimeError(f"Expected {world_size} CUDA devices, got {len(devices)}")
    if len(set(devices)) != 1:
        raise RuntimeError(f"Expected identical GPU models, got {devices}")
    model = devices[0]
    if expected_model is not None and model != expected_model:
        raise RuntimeError(f"Expected GPU model {expected_model}, got {model}")

    subprocess.run(["nvidia-smi", "topo", "-m"], check=False)

    for data_size in data_sizes:
        print(f"\nworld_size={world_size} data_size={data_size}", flush=True)
        subprocess.run(
            [
                python,
                "-u",
                "cs336_systems/bench_distributed.py",
                "--world_size",
                str(world_size),
                "--data_size",
                str(data_size),
            ],
            cwd=REMOTE_PROJECT,
            check=True,
        )

    return model


@app.local_entrypoint()
def main(world_sizes: str = "2,4,6", data_sizes: str = "1,10,100,1000", gpu: str = DEFAULT_GPU):
    worlds = parse_sizes(world_sizes, "--world-sizes")
    sizes = parse_sizes(data_sizes, "--data-sizes")
    if any(world not in WORLD_SIZES for world in worlds):
        raise ValueError("--world-sizes accepts 2, 4, and 6")
    gpu = gpu.strip().upper()
    if not gpu or ":" in gpu or "," in gpu:
        raise ValueError("--gpu takes one GPU model, for example H200")

    if gpu == "A100":
        gpu = DEFAULT_GPU

    model = "NVIDIA A100-SXM4-80GB" if gpu == DEFAULT_GPU else None
    for world in worlds:
        model = run.with_options(gpu=f"{gpu}:{world}").remote(world, sizes, model)
