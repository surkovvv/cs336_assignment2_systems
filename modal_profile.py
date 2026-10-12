"""Infrastructure only: execute the existing CS336 profiler on Modal."""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import modal

PROJECT = Path(__file__).resolve().parent
REMOTE_PROJECT = Path("/workspace/assignment2-systems")
VOLUME_NAME = "cs336-profile-results"
SIZE_NAMES = ("small", "medium", "large", "xl", "10B")


def build_image():
    image = modal.Image.debian_slim(python_version="3.13").uv_pip_install("uv==0.12.8")
    if not modal.is_local():
        return image
    basics = PROJECT.parent / "stanford-cs336-assignment1-basics"
    for local, remote in ((PROJECT, REMOTE_PROJECT), (basics, REMOTE_PROJECT.parent / basics.name)):
        for name in ("pyproject.toml", "README.md"):
            image = image.add_local_file(local / name, str(remote / name), copy=True)
        package = "cs336_systems" if local == PROJECT else "cs336_basics"
        image = image.add_local_dir(
            local / package, str(remote / package), copy=True,
            ignore=["**/__pycache__/**", "**/*.pyc"],
        )
    return (
        image.add_local_file(PROJECT / "uv.lock", str(REMOTE_PROJECT / "uv.lock"), copy=True)
        .run_commands(f"cd {REMOTE_PROJECT} && uv sync --frozen --no-dev")
        .env({"PYTHONUNBUFFERED": "1", "WANDB_MODE": "disabled"})
    )


app = modal.App("cs336-profile")
image = build_image()
results_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image, gpu="B200", cpu=4, memory=32768,
    timeout=1800, scaledown_window=2, max_containers=1,
    volumes={"/results": results_volume},
)
def run_benchmarks(jobs: list[dict], regime: int, w_steps: int, n_steps: int, run_id: str, precision: str = "fp32"):
    """Each model gets its own Python process; timing code stays in profile_code.py."""
    import csv

    destination = Path("/results") / run_id
    destination.mkdir(parents=True, exist_ok=False)
    python = REMOTE_PROJECT / ".venv/bin/python"
    probe = subprocess.run(
        [str(python), "-c", (
            "import json,sys,torch,cs336_basics,cs336_systems; "
            "assert torch.cuda.is_available(), 'CUDA unavailable'; "
            "print(json.dumps(dict(python=sys.version,torch=torch.__version__,"
            "cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),"
            "gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,"
            "basics=cs336_basics.__file__,systems=cs336_systems.__file__)))"
        )], cwd=REMOTE_PROJECT, check=True, capture_output=True, text=True,
    )
    environment = json.loads(probe.stdout.strip().splitlines()[-1])
    print(json.dumps(environment, indent=2), flush=True)
    metadata = {
        "run_id": run_id, "environment": environment, "precision": precision,
        "regime": regime, "w_steps": w_steps, "n_steps": n_steps,
        "profile_sha256": hashlib.sha256(
            (REMOTE_PROJECT / "cs336_systems/profile_code.py").read_bytes()
        ).hexdigest(),
        "lock_sha256": hashlib.sha256((REMOTE_PROJECT / "uv.lock").read_bytes()).hexdigest(),
        "jobs": [],
        "note": "Timing columns are emitted unchanged by the student's profiler.",
    }
    combined_rows = []
    script = "profile_code_bf16.py" if precision == "bf16" else "profile_code.py"
    metadata["entrypoint"] = script
    metadata["entrypoint_sha256"] = hashlib.sha256(
        (REMOTE_PROJECT / "cs336_systems" / script).read_bytes()
    ).hexdigest()
    for job in jobs:
        size, config = job["size"], job["config"]
        config_path = destination / f"{size}.jsonl"
        config_path.write_text(json.dumps(config) + "\n")
        command = [
            str(python), f"cs336_systems/{script}",
            "--input", str(config_path), "--regime", str(regime),
            "--w_steps", str(w_steps), "--n_steps", str(n_steps),
            "--output", str(destination / size),
        ]
        print(f"\nRunning {size}: precision={precision}, regime={regime}, warmup={w_steps}, steps={n_steps}", flush=True)
        with (destination / f"{size}.log").open("w") as log:
            process = subprocess.Popen(
                command, cwd=REMOTE_PROJECT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            exit_code = process.wait()
        job_result = {"size": size, "config": config, "exit_code": exit_code}
        metadata["jobs"].append(job_result)
        if exit_code == 0:
            with (destination / f"{size}.csv").open(newline="") as source:
                rows = list(csv.DictReader(source))
            if len(rows) != 1:
                job_result["error"] = f"Expected one result row, received {len(rows)} in {size}.csv"
                print(job_result["error"], flush=True)
            else:
                combined_rows.extend({"size": size, "precision": precision, **row} for row in rows)
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        results_volume.commit()
        if job_result.get("error"):
            break
        if exit_code:
            print(f"{size} failed; log saved. Continuing with remaining sizes.", flush=True)
    if combined_rows:
        with (destination / "combined.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(combined_rows[0]))
            writer.writeheader()
            writer.writerows(combined_rows)
    results_volume.commit()
    return {
        "metadata": metadata,
        "files": {path.name: path.read_bytes() for path in destination.iterdir() if path.is_file()},
    }


@app.local_entrypoint()
def main(
    sizes: str = "small", regime: int = 2, w_steps: int = 5, n_steps: int = 10,
    input: str = "benches/model_configs.jsonl", output: str = "benches/modal_runs",
    precision: str = "fp32",
):
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    if regime not in (0, 1, 2) or w_steps < 0 or n_steps < 2:
        raise ValueError("Use regime 0/1/2, w_steps >= 0, n_steps >= 2 (standard deviation).")
    configs = [json.loads(line) for line in (PROJECT / input).read_text().splitlines() if line.strip()]
    if len(configs) != len(SIZE_NAMES):
        raise ValueError("Expected five configurations in table order: small, medium, large, xl, 10B.")
    selected = list(SIZE_NAMES) if sizes.lower() == "all" else [s.strip() for s in sizes.split(",")]
    if not selected or len(set(selected)) != len(selected) or any(s not in SIZE_NAMES for s in selected):
        raise ValueError(f"Choose all or comma-separated names from {SIZE_NAMES}.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + precision + "-" + uuid4().hex[:8]
    local_output = PROJECT / output / run_id
    local_output.mkdir(parents=True, exist_ok=False)
    jobs = [{"size": s, "config": configs[SIZE_NAMES.index(s)]} for s in selected]
    (local_output / "request.json").write_text(json.dumps({
        "run_id": run_id, "jobs": jobs, "regime": regime, "w_steps": w_steps, "n_steps": n_steps, "precision": precision,
    }, indent=2) + "\n")
    print(f"Local results: {local_output}")
    print(f"Persistent results: Modal Volume {VOLUME_NAME}/{run_id}", flush=True)
    response = run_benchmarks.remote(jobs, regime, w_steps, n_steps, run_id, precision)
    for name, content in response["files"].items():
        (local_output / name).write_bytes(content)
    print(f"Downloaded results to {local_output}")
    failures = [job["size"] for job in response["metadata"]["jobs"] if job["exit_code"] or job.get("error")]
    if failures:
        raise RuntimeError(f"Failed models: {', '.join(failures)}. See downloaded logs.")
