# Run the existing profiler on Modal

Run these commands from the `assignment2-systems` directory. Modal is already installed in your local virtual environment and authenticated with the `surkovvv` profile.

## Short infrastructure check

```sh
uv run modal run --profile surkovvv modal_profile.py --sizes small --regime 2 --w-steps 1 --n-steps 2
```

This check verifies the GPU, your two packages, the existing profiler CLI, persistent storage, and downloading outputs. Its timings are not the assignment's final measurements.

## Assignment 2.1.3 measurements

The launcher defaults to 5 warmup steps and 10 measured steps. Start with one size:

```sh
uv run modal run --profile surkovvv modal_profile.py --sizes small --regime 2
```

Run all five table configurations:

```sh
uv run modal run --profile surkovvv modal_profile.py --sizes all --regime 2
```

Select a subset with `--sizes small,medium,large`. Modes follow your profiler: 0 = forward, 1 = forward/backward, 2 = forward/backward/optimizer. The CLI uses `--w-steps` and `--n-steps` with hyphens; the launcher translates them to your profiler's underscore arguments.

For the warmup comparison in 2.1.3(c), rerun the same selected sizes, mode, and measurement count with `--w-steps 0`, `--w-steps 1`, `--w-steps 2`, and `--w-steps 5`.

## Mixed precision with BF16 autocast

First check one small model with forward/backward, one warmup step, and two measured steps:

```sh
uv run modal run --profile surkovvv modal_profile.py --sizes small --regime 1 --use-autocast --w-steps 1 --n-steps 2
```

Then run all five configurations with the default 5 warmup steps and 10 measured steps:

```sh
uv run modal run --profile surkovvv modal_profile.py --sizes all --regime 1 --use-autocast
```

`--use-autocast` is a flag: do not append `True` or `False`. Omit it (or use `--no-use-autocast`) for the FP32 baseline. Keep `--precision fp32` (the default) for the autocast experiment: `--precision bf16` selects the separate full-BF16 entrypoint, which does not accept `--autocast`. Do not combine it with `--use-autocast`.

The launcher forwards `use_autocast` to the remote function, which conditionally adds `--autocast` to the profiler command. The profiler parses that flag with `action="store_true"`.

### Current profiler issues to resolve before measuring

The following issues were found in the profiler on 2026-10-27. A correct launch command alone does not resolve them:

- `AttributeError: 'Namespace' object has no attribute 'batch_size'`: the argument is named `--bs`, so argparse stores it as `bs`. The final call to `bench_all` must read that attribute, or the argument declaration must explicitly set its destination to `batch_size`.
- `bench_all` receives `autocast` but its call to `run_profile_timeit` currently omits it. Forward the value explicitly; otherwise the timing function keeps its default `False`, even when the command includes the flag.
- The autocast context currently receives `batch.device` as `device_type`. PyTorch expects a string such as `"cuda"`; the tensor device's `type` attribute supplies that string.

Warmup currently calls the model outside the autocast context. Account for that when reviewing the experiment's warmup behavior.

The launcher records the requested autocast flag in `request.json` and `metadata.json` under `autocast`. The run ID and `combined.csv` precision column still say `fp32` for both the baseline and autocast runs: use the metadata to distinguish them. These fields record the request, so they do not prove that the profiler actually entered autocast.

## Full BF16 experiment

To run only 10B with BF16 weights:

```sh
uv run modal run --profile surkovvv modal_profile.py --sizes 10B --regime 2 --precision bf16
```

For a short check, append `--w-steps 1 --n-steps 2`. For comparisons at the same precision, use `--sizes all --precision bf16`.

This selects `cs336_systems/profile_code_bf16.py`, a small entrypoint that reuses your current profiler and initializes the model with `dtype=torch.bfloat16`. In your current AdamW implementation, gradients and both moments also use BF16; there are no FP32 master weights. RMSNorm and other explicitly FP32 internal operations retain their existing behavior. This is full BF16 rather than FP32 weights with BF16 autocast, and it changes numerical behavior.

Weights, gradients, and both moments together require approximately 95.61 GiB for your 12.83B-parameter table configuration, excluding activations and temporary tensors. Fitting the complete workload must still be verified on GPU.

BF16 CSV/LaTeX files have a `precision` column. Each model also writes `.memory.json` with peak allocated/reserved GPU memory and the precision policy. Modal run IDs and metadata record the chosen precision; the default remains FP32. Timing logic is reused unchanged, including any diagnostic prints in your profiler.

## Outputs

Each invocation creates a unique directory under `benches/modal_runs/`. Files include per-model `.csv`, `.tex`, `.jsonl`, and `.log`; `combined.csv` adds a model-size column to successful results. `metadata.json` records GPU, available GPU memory, Python, PyTorch, CUDA, package import locations, profiler/lock hashes, parameters, and exit codes. `request.json` records the local request even if the remote call fails.

The same remote files persist in the Modal Volume `cs336-profile-results` under the printed run ID. If the local connection fails, retrieve a result manually, replacing RUN_ID:

```sh
uv run modal volume get --profile surkovvv cs336-profile-results /RUN_ID/small.csv benches/recovered_small.csv
```

The launcher uploads only the project/package files it needs. It does not upload your `.venv`, datasets, training checkpoints, or local credentials. The image installs dependencies from the existing `uv.lock` using `uv sync --frozen --no-dev`, with both repositories at matching relative paths.

## GPU and failure handling

The launcher currently requests one B200, four CPU cores, 32 GiB host RAM, and a 30-minute function timeout. Host RAM is distinct from GPU VRAM. The B200 gives more VRAM for large FP32 runs, but fitting the 10B full-training case is not guaranteed; activations, optimizer state, and temporary allocations also occupy VRAM.

Each size runs in a separate subprocess. A failed model's log is preserved and other selected sizes are attempted. If the profiler exits successfully but writes an empty or unexpected CSV, the launcher stops the sequence and reports that output validation failed. It never invents timing rows or changes their units.

GPU containers scale down after two idle seconds; the app runs on demand via `modal run`. There is no permanent deployment or scheduled job.

## Scope

This file covers hosting and executing your existing end-to-end benchmark. It does not change the model, optimizer, timing logic, or assignment answers. A successful infrastructure check does not certify that the profiler's measurements are correct: review CUDA synchronization and output units separately for the assignment.

Nsight Systems (2.1.4) is a separate profiling environment; `nsys` is not installed by this launcher. Profiler traces should be collected in separate runs from the end-to-end timing measurements.

## Shared environment and simple wrappers

`modal_common.py` defines the shared project paths, image, and script runner. `modal_profile.py` imports this environment; `modal_nsys.py` extends the same image with Nsight Systems.

To run the existing mixed-precision toy script and stream its prints without benchmark arguments:

```sh
uv run modal run --profile surkovvv modal_mixed_precision.py
```

This wrapper uses CPU because the toy script currently creates CPU tensors. To make another wrapper, copy it, choose a distinct Modal app name, and replace the project-relative script path passed to `run_script`. Add a GPU requirement to the function decorator if the script needs CUDA. The runner uses the project's virtual environment, streams stdout/stderr, and raises if the script fails.
