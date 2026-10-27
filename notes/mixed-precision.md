Problem (mixed_precision_accumulation):
results:
    tensor(10.0001, device='mps:0')
    tensor(9.9531, device='mps:0', dtype=torch.float16)
    tensor(9.9531, device='mps:0', dtype=torch.float16)
    tensor(9.9531, device='mps:0', dtype=torch.float16)

As we can see, original result with fp32/fp32 combination is different from any other attempt. What's interesting - each of fp16 casting result gives us the same result(on my mps)

Problem (bencmarking_mixed_precision):
a)
For given model, 
model params: fp32
output of first fc: fp16
output of layer norm: fp32
model's logits: fp16(cuz logits = fc2 output)
the loss: depend on 
a) under autocast context or outside of it
b) some functions are casting to fp32(like cross_entropy)
model's grads(.grad): fp32 - same as params dtype

b) 
I guess var(x) is the most sensible because of fp16 lower range. If we'll use bf16 instead, I guess, we can try not to upcast it to fp32. Just because the range of bf16 is the same as in fp32.

c)
| Size | Forward mean | Forward std | Backward mean | Backward std | Forward + backward mean | Forward + backward std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 27.159 | 36.018 | 29.767 | 24.503 | 56.926 | 60.492 |
| medium | 42.205 | 31.617 | 51.441 | 20.667 | 93.646 | 52.242 |
| large | 58.884 | 30.873 | 74.426 | 23.839 | 133.310 | 54.545 |
| xl | 187.881 | 395.326 | 99.234 | 23.650 | 287.115 | 400.108 |
| 10B | 243.346 | 290.182 | 272.296 | 34.142 | 515.642 | 308.420 |

Source: [combined.csv](../benches/modal_runs/20261027T185100Z-fp32-8d091ad2/combined.csv).

### Comparison with FP32

| Model | FP32 forward + backward (ms) | BF16 mixed precision forward + backward (ms) | Speedup (FP32 / mixed) | Time reduction |
|:---|---:|---:|---:|---:|
| small | 55.49 | 56.93 | 0.97× | −2.6% |
| medium | 133.74 | 93.65 | 1.43× | 30.0% |
| large | 304.47 | 133.31 | 2.28× | 56.2% |
| xl | 760.15 | 287.12 | 2.65× | 62.2% |
| 10B | — | 515.64 | — | — |

Compared with the FP32 baseline in [benchmark_report_fp32_bf16.md](benchmark_report_fp32_bf16.md), the observed forward-plus-backward speedup from BF16 mixed precision increases with model size, from 1.43× for medium to 2.28× for large and 2.65× for XL.
The small model shows no clear benefit, with a 2.6% increase in mean execution time that is small relative to the measured variability.
XL and 10B show unusually large timing variability, especially in the forward pass, so repeated measurements are needed to confirm the trend; no FP32 speedup is reported for 10B because the baseline table has no corresponding result.
