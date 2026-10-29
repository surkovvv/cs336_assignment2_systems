"""Parameter-only entrypoint for the existing profiler under Nsight Systems."""

from argparse import ArgumentParser
import torch
from cs336_systems import profile_code as profiler


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--regime", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--w_steps", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    args = parser.parse_args()
    if args.bs < 1 or args.context_length < 1 or args.w_steps < 0 or args.n_steps < 2:
        parser.error("positive batch/context, nonnegative warmup, and at least two measurements required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    original = profiler.TransformerLM
    dtype = torch.float32 if args.precision == "fp32" else torch.bfloat16

    def make_model(**kwargs):
        kwargs["dtype"] = dtype
        model = original(**kwargs)
        forward = model.forward
        call = 0

        def labeled_forward(*inputs, **options):
            nonlocal call
            label = "warmup/forward" if call < args.w_steps else f"step_{call - args.w_steps}/forward"
            call += 1
            with torch.cuda.nvtx.range(label):
                return forward(*inputs, **options)

        model.forward = labeled_forward
        return model

    profiler.TransformerLM = make_model
    try:
        profiler.bench_all(
            args.input, args.w_steps, args.n_steps, args.regime, args.output,
            batch_size=args.bs, context_length=args.context_length,
        )
    finally:
        profiler.TransformerLM = original


if __name__ == "__main__":
    main()
