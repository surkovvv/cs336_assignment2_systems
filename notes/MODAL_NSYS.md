# Nsight Systems on Modal

From the assignment2-systems directory:

```sh
uv run modal run --profile surkovvv modal_nsys.py
```

Defaults: FP32, B200, batch size 1, regime 2, 5 warmup steps, 10 measured steps.

| Model | Context lengths |
|---|---|
| small | 128, 512, 8192 |
| xl | 128, 512, 2048 |

Each combination runs in a separate process under `nsys profile`. Software CUDA tracing (`cuda-sw`), NVTX and OS runtime tracing are enabled. The default Blackwell hardware trace produced no GPU kernel events in the Modal check, so software tracing is selected explicitly. CPU sampling and context-switch tracing are disabled for container compatibility. Nsight's PyTorch integration adds forward, backward and optimizer labels; the entrypoint additionally labels forward calls as `warmup/forward` or `step_N/forward`.

Reports include initialization and warmup. Select measured `step_N/forward` ranges when inspecting forward kernels; whole-report summaries also include initialization, warmup, backward and optimizer work. Profiling adds overhead, so use ordinary benchmark runs for timing comparisons.

The `.nsys-rep`, `.sqlite`, kernel/API/NVTX summaries, profiler CSV/LaTeX, input configs and logs are downloaded to `benches/nsys_runs/<run_id>/`. Reports also persist on Volume `cs336-profile-results`. The launcher verifies that every successful trace contains CUDA kernel events.

Open `.nsys-rep` in the NVIDIA Nsight Systems desktop application, version 2026.5.1 or newer. Local profiling installation is unnecessary for collection.

To collect one short trace first:

```sh
uv run modal run --profile surkovvv modal_nsys.py --sizes small --contexts 128 --w-steps 1 --n-steps 2
```

Other choices: `--sizes xl`, `--contexts 512,2048`, `--precision bf16`, `--regime 0`. The context override applies to every selected model. BF16 uses the existing full-BF16 precision policy, including BF16 AdamW moments.

To recover an interrupted download, replace RUN_ID:

```sh
uv run modal volume get --profile surkovvv cs336-profile-results /RUN_ID benches/nsys_runs/recovered_RUN_ID
```

The existing model and profiler are reused. `profile_code_nsys.py` supplies batch/context arguments directly to `bench_all`; timing logic is unchanged. The regular Modal benchmarking launcher is separate from this Nsight launcher.
