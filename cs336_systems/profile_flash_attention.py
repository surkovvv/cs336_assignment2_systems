from itertools import product
from triton.testing import do_bench
import torch
from flash_attention2 import TritonFlashAttention
from cs336_basics.transformer import sdpa
import argparse


def bench_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False, mode: str = "default"):
    if mode == "default":
        forward_foo = sdpa
        seq_len = Q.size(-2)
        upper_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device), diagonal=1
        )
        mask_arg = ~upper_mask if is_causal else None
    else:
        forward_foo = TritonFlashAttention.apply
        mask_arg = is_causal
    
    def call_forward():
        res = forward_foo(Q, K, V, mask_arg)
        return res

    return call_forward


def bench_backward(Q, K, V, is_causal=False, mode="default"):
    call_forward = bench_forward(Q, K, V, is_causal, mode)

    O = call_forward()
    dO = torch.randn_like(O)

    def call_backward():
        O.backward(dO, retain_graph=True)

    return call_backward


def bench_forward_backward(Q, K, V, is_causal=False, mode="default"):
    call_forward = bench_forward(Q, K, V, is_causal, mode)

    dO = torch.randn_like(Q)

    def call_forward_backward():
        O = call_forward()
        O.backward(dO)

    return call_forward_backward


def run_benchmarks(bench_factory):
    seq_lens = [2**p for p in range(7, 17)]
    d_models = [16, 32, 64, 128]
    precisions = [torch.bfloat16, torch.float32]

    results = []

    for seq_len, d_model, dtype in product(seq_lens, d_models, precisions):
        Q, K, V = [
            torch.randn(
                (1, seq_len, d_model),
                device="cuda",
                dtype=dtype,
                requires_grad=True,
            )
            for _ in range(3)
        ]

        for mode in ("default", "triton"):
            call = bench_factory(
                Q, K, V,
                is_causal=True,
                mode=mode,
            )

            latency_ms = do_bench(
                call,
                warmup=25,
                rep=100,
                return_mode="median",
                grad_to_none=[Q, K, V],
            )

            results.append({
                "regime": bench_factory.__name__,
                "seq_len": seq_len,
                "d_model": d_model,
                "precision": str(dtype),
                "mode": mode,
                "latency_ms": latency_ms,
            })

            print(
                f"N={seq_len:5d} | D={d_model:3d} | "
                f"{str(dtype):14s} | {mode:7s} | "
                f"{latency_ms:.4f} ms"
            )

            del call
            Q.grad = K.grad = V.grad = None

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regime",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="0: forward-only; 1: backward-only; 2: forward + backward",
    )
    args = parser.parse_args()

    factories = {
        0: bench_forward,
        1: bench_backward,
        2: bench_forward_backward,
    }

    results = run_benchmarks(factories[args.regime])
