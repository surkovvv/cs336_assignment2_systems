"""Run the unchanged profiler: uv run --with modal modal run modal_attention.py."""

import subprocess
from datetime import datetime, timezone
from uuid import uuid4

import modal

from modal_common import PROJECT, REMOTE_PROJECT, image

app = modal.App("cs336-attention")


@app.function(
    image=image.apt_install("build-essential"),
    gpu=["H200", "H100"],
    cpu=4,
    memory=32768,
    timeout=1800,
    scaledown_window=2,
    max_containers=1,
)
def run(use_torch_compile: bool = False) -> dict:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
        check=True, capture_output=True, text=True,
    ).stdout
    print(gpu, end="", flush=True)
    mode = f"use_torch_compile={use_torch_compile}\n"
    print(mode, end="", flush=True)
    lines = [gpu, mode]
    # Use the existing environment and execute __main__, just like a local run.
    command = [str(REMOTE_PROJECT / ".venv/bin/python"), "-u", "cs336_systems/profile_attention.py"]
    if use_torch_compile:
        command.append("--use_torch_compile")
    with subprocess.Popen(
        command,
        cwd=REMOTE_PROJECT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        exit_code = process.wait()
    return {"log": "".join(lines), "exit_code": exit_code}


@app.local_entrypoint()
def main(output: str = "benches/modal_attention_runs", use_torch_compile: bool = False):
    response = run.remote(use_torch_compile=use_torch_compile)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    destination = PROJECT / output / f"{run_id}.log"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(response["log"])
    print(f"Saved output to {destination}")
    if response["exit_code"]:
        raise RuntimeError(f"Profiler exited with code {response['exit_code']}. See {destination}")
