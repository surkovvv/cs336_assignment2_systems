"""Run the FlashAttention profiler on one Modal B200.

Usage: uv run --with modal modal run modal_flash_attention.py --regime 0
"""

import subprocess
from datetime import UTC, datetime
from uuid import uuid4

import modal

from modal_common import PROJECT, REMOTE_PROJECT, image

app = modal.App("cs336-flash-attention")


@app.function(
    image=image.apt_install("build-essential"),
    gpu="H200",
    cpu=4,
    memory=32768,
    timeout=3600,
    scaledown_window=2,
    max_containers=1,
)
def run(regime: int = 0) -> dict:
    if regime not in (0, 1, 2):
        raise ValueError("regime must be 0, 1, or 2")

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    header = f"{gpu}regime={regime}\n"
    print(header, end="", flush=True)
    lines = [header]

    command = [
        str(REMOTE_PROJECT / ".venv/bin/python"),
        "-u",
        "cs336_systems/profile_flash_attention.py",
        "--regime",
        str(regime),
    ]
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
def main(regime: int = 0, output: str = "benches/modal_flash_attention_runs"):
    if regime not in (0, 1, 2):
        raise ValueError("regime must be 0, 1, or 2")

    response = run.remote(regime=regime)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-regime{regime}-" + uuid4().hex[:8]
    destination = PROJECT / output / f"{run_id}.log"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(response["log"], encoding="utf-8")
    print(f"Saved output to {destination}")

    if response["exit_code"]:
        raise RuntimeError(f"Profiler exited with code {response['exit_code']}. See {destination}")
