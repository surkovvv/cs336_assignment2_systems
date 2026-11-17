from cs336_basics.transformer import sdpa
import torch
from timeit import default_timer as dt
import itertools
import statistics
from argparse import ArgumentParser

def bench_attention(
    use_torch_compile: bool = False,
    batch_size: int = 8,
    d_models: list[int] = [16, 32, 64, 128],
    seq_lens: list[int] = [256, 1024, 4096, 8192, 16384],
    n_runs: int = 100,
    device: torch.device = torch.device("cuda")
):
    n_warmup_runs = 5
    timings = {}

    for d_model, seq_len in itertools.product(d_models, seq_lens):
        try:
            Q = torch.randn((batch_size, seq_len, d_model), requires_grad=True, device=device)
            K = torch.randn((batch_size, seq_len, d_model), requires_grad=True, device=device)
            V = torch.randn((batch_size, seq_len, d_model), requires_grad=True, device=device)

            
            attention_forward = torch.compile(sdpa) if use_torch_compile else sdpa

            for _ in range(n_warmup_runs):
                res = attention_forward(Q, K, V)
                dO = torch.ones_like(res)
                res.backward(dO)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            forward_times = []
            backward_times = []

            memory_allocated = 0

            for i in range(n_runs):
                time_before_forward = dt()
                res = attention_forward(Q, K, V)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                time_after_forward = dt()
                forward_times.append(time_after_forward - time_before_forward)

                memory_allocated = max(memory_allocated, torch.cuda.memory_allocated(device=device))
                print(f"Memory before {i + 1}th backward:", memory_allocated, " bytes")

                dO = torch.ones_like(res)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                time_before_backward = dt()
                res.backward(dO)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                time_after_backward = dt()
                backward_times.append(time_after_backward - time_before_backward)

            timings[(d_model, seq_len)] = (forward_times, backward_times, memory_allocated)

        except torch.cuda.OutOfMemoryError as e:
            print(f"For combination d_model={d_model}, seq_len={seq_len} there was an cudaOutOfMemory error:", e)

    return timings


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--use_torch_compile", action="store_true", default=False, required=False)
    args = parser.parse_args()

    timings = bench_attention(args.use_torch_compile)

    for (d_model, seq_len) in timings.keys():
        forward_times, backward_times, memory_allocated = timings[(d_model, seq_len)]
        print(f"For combination d_model={d_model}, seq_len={seq_len}:")
        print("  forward time(mean over 100 runs): ", statistics.mean(forward_times))
        print("  backward time(mean over 100 runs): ", statistics.mean(backward_times))
        print("  max memory allocated over 100 runs before forward: ", memory_allocated, " bytes")
