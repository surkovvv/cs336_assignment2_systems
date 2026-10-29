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
