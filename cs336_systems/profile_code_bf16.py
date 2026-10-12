"""Run the existing profiler with BF16 parameters, gradients, and AdamW moments.

This is full BF16, not autocast mixed precision. The model's existing internal
FP32 operations (for example, RMSNorm) remain as implemented.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch

from cs336_systems import profile_code as profiler


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--regime", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--w_steps", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=10)
    args = parser.parse_args()
    if args.w_steps < 0 or args.n_steps < 2:
        parser.error("w_steps must be >= 0; n_steps must be >= 2 for standard deviation")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This benchmark requires a CUDA GPU with BF16 support.")

    original_model = profiler.TransformerLM

    def make_bf16_model(**kwargs):
        kwargs["dtype"] = torch.bfloat16
        model = original_model(**kwargs)
        if any(p.dtype != torch.bfloat16 for p in model.parameters()):
            raise RuntimeError("Some model parameters were not initialized in BF16.")
        print("precision=bf16; parameter_dtype=bfloat16; AdamW moments follow parameter dtype", flush=True)
        return model

    profiler.TransformerLM = make_bf16_model
    torch.cuda.reset_peak_memory_stats()
    try:
        profiler.bench_all(args.input, args.w_steps, args.n_steps, args.regime, args.output)
    finally:
        profiler.TransformerLM = original_model
        torch.cuda.synchronize()
        memory = {
            "precision": "bf16", "parameter_dtype": "bfloat16",
            "gradient_dtype": "bfloat16", "adamw_moment_dtype": "bfloat16",
            "autocast": False, "fp32_master_weights": False,
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        Path(args.output + ".memory.json").write_text(json.dumps(memory, indent=2) + "\n")
        print(json.dumps(memory, indent=2), flush=True)

    results = pd.read_csv(args.output + ".csv")
    results.insert(0, "precision", "bf16")
    results.to_csv(args.output + ".csv", index=False)
    results.to_latex(args.output + ".tex", index=False)


if __name__ == "__main__":
    main()
