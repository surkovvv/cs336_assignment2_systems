# 3: gradient_checkpoiting

## a.

I'd use nested checkpoints covering prefixes: blocks 1 through \(N\), blocks 1 through \(N-1\), and so on down to the first block. During backward, I would recompute each required prefix from the original input, materializing the residuals of its final block and then performing that block’s backward pass. Ignoring checkpoint bookkeeping, peak activation memory consists of the original input plus a constant number of block-sized activations and one block’s internal residuals, giving \(O(1)\) memory with respect to \(N\). This is asymptotically minimal because the original input must remain available for recomputation. The strategy requires \(N+\sum_{k=1}^{N}k\) block-forward evaluations and \(N\) block-backward evaluations, giving \(O(N^2)\) compute.

```python
def first_order(x):
    return blocks[0](x)


def factory(foo, block):
    def calc_block_checkpoint(x):
        x = checkpoint(foo, x)
        return block(x)
    return calc_block_checkpoint


second_order = factory(first_order, blocks[1])
third_order = factory(second_order, blocks[2])
# ... continue building successive prefixes, in loop, ofc
n_order = factory(prev_order, blocks[N - 1])

x = checkpoint(n_order, x)
```

## b.

### Peak memory, GiB

XL model, FP32, batch size 1, context length 2048, NVIDIA H200.

| Block size | Forward only | Full train step |
| --- | ---: | ---: |
| None (no checkpointing) | 15.0 | 102.5 |
| 1 | 15.10 | 54.02 |
| 2 | 15.12 | 55.22 |
| 4 | 15.12 | 57.63 |

The no-checkpointing values are from the earlier measurements. Checkpointing values were reconstructed from memory snapshots using active allocator block sizes, including allocator rounding.

### Peak memory, batch size 4, GiB

XL model, FP32, context length 2048, NVIDIA H200. Full train step only (regime 2).

| Block size | Full train step |
| --- | ---: |
| None (no checkpointing) | OOM |
| 1 | 63.66 |
| 2 | 69.63 |
| 4 | 81.55 |

Checkpointing peaks were reconstructed from memory snapshots using the same method as above.

Since nested checkpoint calls are not allowed, I divide the model into consecutive blocks of layers and checkpoint each block separately. With a block size of 1, we retain 32 block inputs and need to recompute the internal activations of only one layer at a time; with block sizes of 2 and 4, we retain 16 and 8 block inputs, respectively, but need the internal activations of two or four layers during recomputation. My hypothesis was that a block size of 1 would minimize peak memory because a transformer layer’s internal activations would be substantially more expensive to retain than its input tensor, whose shape is 4 x 2048 x 2560 in this configuration. Rather than calculating the exact memory required for all activations needed by backward, I tested this hypothesis by profiling. The measured peaks were **63.66 GiB**, **69.63 GiB**, and **81.55 GiB** for block sizes **1**, **2**, and **4**, respectively, while the run without checkpointing ran out of memory, confirming that **block size 1 was best among the tested configurations**.