# Notes for nsys part

### Before all:

1) I'll choose small and xl model sizes. I've checked via code in cs33-basics.training_utils.py that..

- small model will take around 2Gb for weights, grads and optimizer states for fp32 and around 79Gb for the activations for max context_len = 8192(max for B200 and ~180Gb of memory)

- xl model will take around 52Gb for weights, grads and optimizer states for fp32 and around 47Gb for the activations for max context_len = 2048.

My approx numbers:

Number of params for small model:  128.625408 M params
Memory needed for small gpt2 params(weight + grad + optimizer), fp32, context_len=128:  1962.66796875  Mbytes
Memory needed for small gpt2 activations, fp32,  context_len=128:  0.12818146124482155  GB
Memory needed for small gpt2 activations, fp32,  context_len=512:  0.7236633338034153  GB
Memory needed for small gpt2 activations, fp32,  context_len=1024:  2.0098266638815403  GB
Memory needed for small gpt2 activations, fp32,  context_len=2048:  6.26965332403779  GB
Memory needed for small gpt2 activations, fp32,  context_len=4096:  21.53930664435029  GB
Memory needed for small gpt2 activations, fp32,  context_len=8192:  79.07861328497529  GB
Memory needed for small gpt2 activations, fp32,  context_len=16384:  302.1572265662253  GB

Number of params for xl model:  3406.8096 M params
Memory needed for xl gpt2 params(weight + grad + optimizer), fp32, context_len=128:  51983.7890625  Mbytes
Memory needed for xl gpt2 activations, fp32,  context_len=128:  1.0684890784323215  GB
Memory needed for xl gpt2 activations, fp32,  context_len=512:  5.773956302553415  GB
Memory needed for xl gpt2 activations, fp32,  context_len=1024:  15.54791260138154  GB
Memory needed for xl gpt2 activations, fp32,  context_len=2048:  47.09582519903779  GB
Memory needed for xl gpt2 activations, fp32,  context_len=4096:  158.1916503943503  GB
Memory needed for xl gpt2 activations, fp32,  context_len=8192:  572.3833007849753  GB

### Memory estimates (batch size 4, FP32)

| Model | Parameters (millions) | Weights + gradients + optimizer states (Mbytes) |
|:---|---:|---:|
| small | 128.625408 | 1962.66796875 |
| xl | 3406.8096 | 51983.7890625 |

| Context length | Small activations (GB) | XL activations (GB) |
|---:|---:|---:|
| 128 | 0.5127258338034153 | 4.273956302553415 |
| 512 | 2.8946533240377903 | 23.09582519903779 |
| 1024 | 8.03930664435029 | 62.19165039435029 |
| 2048 | 25.07861328497529 | 188.3833007849753 |
| 4096 | 86.15722656622529 | 632.7666015662253 |
| 8192 | 316.3144531287253 | 2289.5332031287253 |
| 16384 | 1208.6289062537253 | 8675.066406253725 |

### (a) Forward-pass timing

#### FP32 results (batch size 4)

| config | forward_times_mean | forward_times_std | backward_times_mean | backward_times_std | optimizer_times_mean | optimizer_times_std | forward+backward_times_mean | forward+backward_times_std | forward+backward+optimizer_times_mean | forward+backward+optimizer_times_std |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small_seq128_bs4_fp32 | 31.042957 | 12.089597 | 51.992844 | 26.624167 | 22.950870 | 12.563190 | 83.035801 | 29.517057 | 105.986671 | 41.380804 |
| small_seq512_bs4_fp32 | 31.257433 | 4.369275 | 53.863613 | 22.714984 | 20.833527 | 3.115038 | 85.121046 | 23.171400 | 105.954573 | 25.838087 |
| small_seq4096_bs4_fp32 | 264.964883 | 6.064642 | 285.520642 | 104.869586 | 282.897028 | 90.197066 | 550.485525 | 110.851423 | 833.382554 | 20.666347 |
| xl_seq128_bs4_fp32 | 143.557828 | 3.350985 | 166.067098 | 55.405439 | 87.987998 | 19.107851 | 309.624926 | 54.042296 | 397.612923 | 71.144687 |
| xl_seq512_bs4_fp32 | 317.691778 | 16.451605 | 456.718708 | 58.449301 | 188.869304 | 12.007225 | 774.410486 | 58.561036 | 963.279790 | 47.124305 |
| xl_seq1024_bs4_fp32 | 602.919144 | 12.381229 | 939.449318 | 100.542163 | 302.117571 | 79.811720 | 1542.368462 | 101.267694 | 1844.486033 | 23.933857 |

#### Comparison with the Python standard-library measurements (context length 512)

Times are in milliseconds, reported as mean ± standard deviation.

| Model | Previous run | Profiled run | Increase (ms) | Increase (%) |
|:---|---:|---:|---:|---:|
| small | 21.63 ± 3.99 | 31.257433 ± 4.369275 | 9.63 | 44.5% |
| xl | 303.35 ± 25.47 | 317.691778 ± 16.451605 | 14.34 | 4.7% |

**Answer.** At context length 512, the profiled forward times are of the same order of magnitude as the earlier Python standard-library measurements, but are higher by 9.63 ms (44.5%) for small and 14.34 ms (4.7%) for xl. My hypothesis is that profiling adds a similar absolute overhead in both cases, which accounts for a larger fraction of the shorter forward pass in the small model.
