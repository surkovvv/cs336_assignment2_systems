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
