"""Run from assignment2-systems with PYTHONPATH="$PWD" uvx modal run <this file>.

Use --gpu H100 to select a different GPU model. The benchmark spawns two ranks.
"""

import subprocess

import modal

from modal_common import image, run_script

app = modal.App("cs336-bench-ddp")
DEFAULT_GPU = "A100-80GB"
WORLD_SIZE = 2


@app.function(
    image=image,
    gpu=f"{DEFAULT_GPU}:{WORLD_SIZE}",
    cpu=4,
    memory=32768,
    timeout=1800,
    scaledown_window=2,
    max_containers=1,
)
def run():
    subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv",
        ],
        check=True,
    )
    subprocess.run(["nvidia-smi", "topo", "-m"], check=False)
    run_script("cs336_systems/bench_ddp.py")


@app.local_entrypoint()
def main(gpu: str = DEFAULT_GPU):
    gpu = gpu.strip().upper()
    if not gpu or ":" in gpu or "," in gpu:
        raise ValueError("--gpu takes one GPU model, for example H100")
    run.with_options(gpu=f"{gpu}:{WORLD_SIZE}").remote()
