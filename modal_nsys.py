"""Collect the requested small/xl context-length matrix with Nsight Systems."""

import hashlib
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import modal
from modal_profile import PROJECT, REMOTE_PROJECT, image as benchmark_image, results_volume

NSYS_VERSION = "2026.5.1.161"
NSYS_DEB = "NsightSystems-linux-cli-public-2026.5.1.161-3889610.deb"
NSYS_URL = "https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/" + NSYS_DEB
image = benchmark_image.apt_install("curl", "ca-certificates").run_commands(
    f"curl -fL --retry 3 {NSYS_URL} -o /tmp/nsys.deb",
    "apt-get update && apt-get install -y --no-install-recommends /tmp/nsys.deb && rm /tmp/nsys.deb",
    "nsys --version",
).add_local_file(PROJECT / "modal_profile.py", "/root/modal_profile.py")
app = modal.App("cs336-nsys")


def make_jobs(sizes, contexts, configs):
    indices = {"small": 0, "xl": 3}
    selected = [s.strip() for s in sizes.split(",")]
    if not selected or len(set(selected)) != len(selected) or any(s not in indices for s in selected):
        raise ValueError("sizes must be small, xl, or small,xl")
    override = [int(s) for s in contexts.split(",")] if contexts else None
    if override and (min(override) < 1 or len(set(override)) != len(override)):
        raise ValueError("contexts must contain distinct positive lengths")
    return [
        {"size": size, "context_length": length, "config": {**configs[indices[size]], "context_length": length}}
        for size in selected
        for length in (override or ([128, 512, 8192] if size == "small" else [128, 512, 2048]))
    ]


@app.function(image=image, gpu="B200", cpu=4, memory=32768, timeout=3600,
              scaledown_window=2, max_containers=1, volumes={"/results": results_volume})
def collect(jobs: list[dict], precision: str, bs: int, regime: int, warmup: int, steps: int, run_id: str,
            capture_api: bool = False):
    destination = Path("/results") / run_id
    destination.mkdir(parents=True, exist_ok=False)
    python = str(REMOTE_PROJECT / ".venv/bin/python")
    probe = subprocess.run([python, "-c", (
        "import json,torch; assert torch.cuda.is_available(); "
        "print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,"
        "gpu=torch.cuda.get_device_name(0),gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory)))"
    )], check=True, capture_output=True, text=True)
    metadata = {
        "run_id": run_id, "environment": json.loads(probe.stdout.strip().splitlines()[-1]),
        "nsys_version": subprocess.check_output(["nsys", "--version"], text=True).strip(),
        "precision": precision, "batch_size": bs, "regime": regime,
        "warmup_steps": warmup, "measurement_steps": steps, "capture_api": capture_api,
        "profiler_sha256": hashlib.sha256((REMOTE_PROJECT / "cs336_systems/profile_code.py").read_bytes()).hexdigest(),
        "jobs": [],
    }
    print(json.dumps(metadata, indent=2), flush=True)
    for job in jobs:
        stem = f"{job['size']}_seq{job['context_length']}_bs{bs}_{precision}"
        config_path = destination / (stem + ".jsonl")
        config_path.write_text(json.dumps(job["config"]) + "\n")
        prefix = destination / stem
        command = [
            "nsys", "profile", "--trace=cuda-sw,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
            "--pytorch=functions-trace,autograd-nvtx", "--force-overwrite=true", "--export=sqlite",
            *(["--capture-range=cudaProfilerApi", "--capture-range-end=stop"] if capture_api else []),
            "--output", str(prefix), "--", python, "cs336_systems/profile_code_nsys.py",
            "--input", str(config_path), "--output", str(prefix), "--bs", str(bs),
            "--context_length", str(job["context_length"]), "--regime", str(regime),
            "--w_steps", str(warmup), "--n_steps", str(steps), "--precision", precision,
        ]
        print(f"Collecting {stem}", flush=True)
        with (destination / (stem + ".log")).open("w") as log:
            proc = subprocess.Popen(command, cwd=REMOTE_PROJECT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = proc.wait()
        result = {**job, "stem": stem, "command": command, "exit_code": code, "kernel_events": 0}
        database = destination / (stem + ".sqlite")
        report = destination / (stem + ".nsys-rep")
        if database.exists():
            with sqlite3.connect(database) as db:
                tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"):
                    if table in tables:
                        result["kernel_events"] += db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        if code == 0 and report.exists() and result["kernel_events"] > 0:
            stats = subprocess.run([
                "nsys", "stats", "--report=cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum",
                "--format=csv", str(database),
            ], capture_output=True, text=True)
            (destination / (stem + ".stats.txt")).write_text(stats.stdout + stats.stderr)
            result["stats_exit_code"] = stats.returncode
            if stats.returncode != 0:
                result["error"] = f"nsys stats failed (exit {stats.returncode}); see {stem}.stats.txt"
        else:
            result["error"] = "Profiling failed, report missing, or no CUDA kernel events captured"
        metadata["jobs"].append(result)
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        results_volume.commit()
    metadata["files"] = [p.name for p in destination.iterdir() if p.is_file()]
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    results_volume.commit()
    return metadata


@app.local_entrypoint()
def main(sizes: str = "small,xl", contexts: str = "", precision: str = "fp32", bs: int = 1,
         regime: int = 2, w_steps: int = 5, n_steps: int = 10, capture_api: bool = False):
    if precision not in ("fp32", "bf16") or bs < 1 or regime not in (0, 1, 2) or w_steps < 0 or n_steps < 2:
        raise ValueError("Invalid precision, batch size, regime, or step count")
    configs = [json.loads(line) for line in (PROJECT / "benches/model_configs.jsonl").read_text().splitlines() if line.strip()]
    if len(configs) != 5:
        raise ValueError("Expected five configs in table order")
    jobs = make_jobs(sizes, contexts, configs)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-nsys-" + precision + "-" + uuid4().hex[:8]
    output = PROJECT / "benches/nsys_runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "request.json").write_text(json.dumps({
        "jobs": jobs, "precision": precision, "bs": bs, "regime": regime,
        "w_steps": w_steps, "n_steps": n_steps, "run_id": run_id, "capture_api": capture_api,
    }, indent=2) + "\n")
    print(f"Results: {output}\nVolume: cs336-profile-results/{run_id}", flush=True)
    metadata = collect.remote(jobs, precision, bs, regime, w_steps, n_steps, run_id, capture_api)
    for name in metadata["files"]:
        with (output / name).open("wb") as target:
            for chunk in results_volume.read_file(f"/{run_id}/{name}"):
                target.write(chunk)
    print(f"Downloaded Nsight reports to {output}")
    failures = [j["stem"] for j in metadata["jobs"] if j.get("error")]
    if failures:
        raise RuntimeError(f"Invalid traces: {failures}. See downloaded logs.")
