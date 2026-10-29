"""Shared Modal environment and script runner for this project."""

import subprocess
from pathlib import Path

import modal

PROJECT = Path(__file__).resolve().parent
REMOTE_PROJECT = Path("/workspace/assignment2-systems")


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
        .add_local_python_source("modal_common", copy=True)
    )


image = build_image()


def run_script(script: str):
    """Run a project-relative script, streaming stdout/stderr and raising on failure."""
    subprocess.run(
        [str(REMOTE_PROJECT / ".venv/bin/python"), "-u", script],
        cwd=REMOTE_PROJECT,
        check=True,
    )
