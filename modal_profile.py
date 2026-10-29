"""Infrastructure only: execute the existing CS336 profiler on Modal."""

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import modal

from modal_common import PROJECT, REMOTE_PROJECT, image

VOLUME_NAME = "cs336-profile-results"
SIZE_NAMES = ("small", "medium", "large", "xl", "10B")




app = modal.App("cs336-profile")
results_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image, gpu="H200", cpu=4, memory=32768,
    timeout=1800, scaledown_window=2, max_containers=1,
    volumes={"/results": results_volume},
)
def run_benchmarks(jobs: list[dict], regime: int, w_steps: int, n_steps: int, run_id: str, precision: str = "fp32", use_autocast: bool = False, profile_memory: bool = False, bs: int = 4):
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
        "regime": regime, "w_steps": w_steps, "n_steps": n_steps, "autocast": use_autocast,
        "profile_memory": profile_memory, "batch_size": bs,
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
        context_length = int(config["context_length"])
        job_name = f"{size}_seq{context_length}_bs{bs}_{precision}"
        job_destination = destination / job_name
        job_destination.mkdir(exist_ok=False)
        config_path = job_destination / "config.jsonl"
        output_prefix = job_destination / "profile"
        snapshot_path = REMOTE_PROJECT / "memory_snapshot.pickle"
        if profile_memory:
            snapshot_path.unlink(missing_ok=True)
        config_path.write_text(json.dumps(config) + "\n")
        command = [
            str(python), f"cs336_systems/{script}",
            "--input", str(config_path), "--regime", str(regime),
            "--w_steps", str(w_steps), "--n_steps", str(n_steps),
            "--output", str(output_prefix),
            "--bs", str(bs), "--context_length", str(context_length),
        ]
        if use_autocast:
            command.append("--autocast")
        if profile_memory:
            command.append("--profile_memory")
        print(f"\nRunning {job_name}: regime={regime}, warmup={w_steps}, steps={n_steps}, profile_memory={profile_memory}", flush=True)
        with (job_destination / "profile.log").open("w") as log:
            process = subprocess.Popen(
                command, cwd=REMOTE_PROJECT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            exit_code = process.wait()
        job_result = {"size": size, "config": config, "batch_size": bs,
                      "output_dir": job_name, "command": command, "exit_code": exit_code}
        if profile_memory:
            if snapshot_path.is_file():
                snapshot = job_destination / "memory_snapshot.pickle"
                shutil.move(str(snapshot_path), str(snapshot))
                job_result["memory_snapshot"] = str(snapshot.relative_to(destination))
            elif exit_code == 0:
                job_result["error"] = "Profiler completed without producing memory_snapshot.pickle"
        metadata["jobs"].append(job_result)
        if exit_code == 0:
            with output_prefix.with_suffix(".csv").open(newline="") as source:
                rows = list(csv.DictReader(source))
            if len(rows) != 1:
                job_result["error"] = f"Expected one result row, received {len(rows)} in {job_name}/profile.csv"
                print(job_result["error"], flush=True)
            else:
                combined_rows.extend({"size": size, "precision": precision, **row} for row in rows)
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        results_volume.commit()
        if job_result.get("error"):
            break
        if exit_code:
            print(f"{job_name} failed; log saved. Continuing with remaining configurations.", flush=True)
    if combined_rows:
        with (destination / "combined.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(combined_rows[0]))
            writer.writeheader()
            writer.writerows(combined_rows)
    results_volume.commit()
    return {
        "metadata": metadata,
        "files": sorted(str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file()),
    }


@app.local_entrypoint()
def main(
    sizes: str = "small", regime: int = 2, w_steps: int = 5, n_steps: int = 10,
    input: str = "benches/model_configs.jsonl", output: str = "benches/modal_runs",
    precision: str = "fp32", use_autocast: bool = False,
    profile_memory: bool = False, contexts: str = "", bs: int = 4
):
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    if regime not in (0, 1, 2) or w_steps < 0 or n_steps < 2 or bs < 1:
        raise ValueError("Use regime 0/1/2, positive bs, w_steps >= 0, n_steps >= 2 (standard deviation).")
    context_lengths = [int(value.strip()) for value in contexts.split(",")] if contexts else None
    if context_lengths and (min(context_lengths) < 1 or len(set(context_lengths)) != len(context_lengths)):
        raise ValueError("contexts must contain distinct positive lengths")
    configs = [json.loads(line) for line in (PROJECT / input).read_text().splitlines() if line.strip()]
    if len(configs) != len(SIZE_NAMES):
        raise ValueError("Expected five configurations in table order: small, medium, large, xl, 10B.")
    selected = list(SIZE_NAMES) if sizes.lower() == "all" else [s.strip() for s in sizes.split(",")]
    if not selected or len(set(selected)) != len(selected) or any(s not in SIZE_NAMES for s in selected):
        raise ValueError(f"Choose all or comma-separated names from {SIZE_NAMES}.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + precision + "-" + uuid4().hex[:8]
    local_output = PROJECT / output / run_id
    local_output.mkdir(parents=True, exist_ok=False)
    jobs = [
        {"size": size, "config": {**configs[SIZE_NAMES.index(size)], "context_length": length}}
        for size in selected
        for length in (context_lengths or [configs[SIZE_NAMES.index(size)]["context_length"]])
    ]
    (local_output / "request.json").write_text(json.dumps({
        "run_id": run_id, "jobs": jobs, "regime": regime, "w_steps": w_steps, "n_steps": n_steps, "precision": precision, "autocast": use_autocast,
        "profile_memory": profile_memory, "batch_size": bs, "contexts": context_lengths,
    }, indent=2) + "\n")
    print(f"Local results: {local_output}")
    print(f"Persistent results: Modal Volume {VOLUME_NAME}/{run_id}", flush=True)
    response = run_benchmarks.remote(jobs, regime, w_steps, n_steps, run_id, precision, use_autocast, profile_memory, bs)
    for name in response["files"]:
        target = local_output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            for chunk in results_volume.read_file(f"/{run_id}/{name}"):
                stream.write(chunk)
    print(f"Downloaded results to {local_output}")
    failures = [job["output_dir"] for job in response["metadata"]["jobs"] if job["exit_code"] or job.get("error")]
    if failures:
        raise RuntimeError(f"Failed configurations: {', '.join(failures)}. See downloaded logs.")
