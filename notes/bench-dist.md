# Single-node all-reduce

All configurations used NVIDIA A100-SXM4-80GB GPUs on Modal, with one NCCL rank per GPU.
New Modal run on [2 GPUs](https://modal.com/apps/surkovvv/main/ap-zQf7IS1kI34xtm69ZpOxPb).
The 4 and 6 GPU results come from the earlier log with five warmup all-reduces.
Column headers show the size of the float32 tensor on each GPU.
Each cell reports the maximum elapsed time across ranks in milliseconds (`default_timer` seconds × 1000).
Each configuration ran five warmup all-reduces, followed by a CUDA synchronization and a rank barrier before one timed all-reduce. CUDA was synchronized again before stopping the timer.

| GPUs | 1 MiB | 10 MiB | 100 MiB | 1000 MiB |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.310 | 0.677 | 0.945 | 6.292 |
| 4 | 0.245 | 0.396 | 1.014 | 7.858 |
| 6 | 0.210 | 0.332 | 1.110 | 8.728 |

At 1 MiB, the maximum rank time ranged from 0.210 to 0.310 ms. At 1000 MiB, the times for two, four, and six GPUs were 6.292, 7.858, and 8.728 ms respectively.

The NVLink topology remains unverified because the topology probe failed in the Modal container.
