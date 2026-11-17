import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from timeit import default_timer as dt

N_WARMUP_STEPS = 5

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '666'
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def run_allreduce(rank, world_size, tensor_size):
    setup(rank, world_size)
    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    data = torch.randn((tensor_size, ), device=device)
    for _ in range(N_WARMUP_STEPS):
        dist.all_reduce(data)
        
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    
    print("First 5 elements of data before all-reduce: ", data[:5])

    dist.barrier(device_ids=[rank] if device != 'cpu' else None)
    time_before = dt()
    dist.all_reduce(data)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    
    time_of_all_reduce = dt() - time_before
    print("First 5 elements of data after all-reduce: ", data[:5])
    print(f"Rank {rank} time of all-reduce op: ", time_of_all_reduce)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", required=True, type=int)
    parser.add_argument("--data_size", required=True, type=int, help="tensors size in MB")
    args = parser.parse_args()
    tensor_size = args.data_size * 1024 ** 2 // 4  # cuz fp32 takes 4 bytes
    mp.spawn(run_allreduce, args=(args.world_size, tensor_size), nprocs=args.world_size)
