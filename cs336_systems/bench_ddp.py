import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_systems.ddp import NaiveDDP
from cs336_systems.bench_distributed import setup
from cs336_basics.transformer import TransformerLM
from cs336_basics.training_utils import AdamW, cross_entropy
from timeit import default_timer as dt


gpt2_xl_config = {
    "vocab_size": 10000,
    "context_length":  512,
    "num_layers" : 32,
    "d_model" : 2560,
    "num_heads" : 32,
    "d_ff" : 10240,
    "theta" : 10000.0
}

NUM_WARMUP_STEPS = 5
NUM_BENCH_STEPS = 100
BATCH_SIZE = 4
USE_SYNC = True


def bench_naive_ddp(rank: int, world_size: int):
    setup(rank, world_size)
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    # init
    model = TransformerLM(**gpt2_xl_config).to(device)
    ddp_model = NaiveDDP(model)
    optimizer = AdamW(ddp_model.parameters(), lr=0.1, betas=(0.9, 0.95), eps=1e-6, weight_decay=0.05)

    assert BATCH_SIZE % world_size == 0, 'BATCH SIZE SHOULD BE DEVIDED BY WORLD SIZE'
    mini_batch_size = BATCH_SIZE // world_size

    # warmup
    for _ in range(NUM_WARMUP_STEPS):
        input_data = torch.randint(0, gpt2_xl_config['vocab_size'], (mini_batch_size, gpt2_xl_config['context_length']), device=device)
        optimizer.zero_grad()
        logits = ddp_model(input_data)
        loss = cross_entropy(logits, input_data)  # same size, for bench it's ok
        loss.backward()
        ddp_model.finish_gradient_synchronization(optimizer)
        optimizer.step()

    if torch.cuda.is_available():
        dist.barrier(device_ids=[rank])
        torch.cuda.synchronize(device)

    e2e_times = []
    communication_times = []

    for _ in range(NUM_BENCH_STEPS):
        # input_data = torch.randint(0, gpt2_xl_config['vocab_size'], (mini_batch_size, gpt2_xl_config['context_length']), device=device)
        start_time = dt()
        optimizer.zero_grad()
        logits = ddp_model(input_data)
        loss = cross_entropy(logits, input_data)  # same size, for bench it's ok(2)
        loss.backward()

        if USE_SYNC and torch.cuda.is_available():
            torch.cuda.synchronize(device)

        time_before_communication = dt()
        ddp_model.finish_gradient_synchronization(optimizer)

        if USE_SYNC and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        
        time_after_communication = dt()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        end_time = dt()

        e2e_times.append((end_time - start_time) * 1000)  # ms
        communication_times.append((time_after_communication - time_before_communication) * 1000)

    e2e_times_tensor = torch.tensor(e2e_times, device=device)
    communication_times_tesnor = torch.tensor(communication_times, device=device)
    proportion_tensor = communication_times_tesnor / e2e_times_tensor

    dist.all_reduce(e2e_times_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(communication_times_tesnor, op=dist.ReduceOp.MAX)
    dist.all_reduce(proportion_tensor, op=dist.ReduceOp.AVG)

    if rank == 0:
        print("E2E times(max among ranks): ", e2e_times_tensor)
        print("Communication times(max among ranks): ", communication_times_tesnor)
        print("Proportion (avg among ranks): ", proportion_tensor)

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2
    mp.spawn(bench_naive_ddp, args=(world_size, ), nprocs=world_size)
