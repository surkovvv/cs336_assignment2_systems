# Transformer benchmarks on NVIDIA B200

Batch size 4, context length 512, vocabulary size 10,000. Unless otherwise noted, each run used 5 warmup steps and 10 measurement steps. Times are in milliseconds, reported as mean ± standard deviation. Total is forward + backward + optimizer, excluding loss computation.

## A. FP32

| Model | Forward | Backward | Forward + backward | Optimizer | Total |
|:---|---:|---:|---:|---:|---:|
| **small** | 21.63 ± 3.99 | 33.86 ± 33.55 | 55.49 ± 33.19 | 13.22 ± 0.78 | 68.70 ± 33.30 |
| **medium** | 60.74 ± 26.21 | 73.00 ± 25.15 | 133.74 ± 34.82 | 42.13 ± 5.58 | 175.87 ± 31.56 |
| **large** | 129.13 ± 41.95 | 175.34 ± 28.04 | 304.47 ± 47.86 | 66.97 ± 9.69 | 371.44 ± 44.25 |
| **xl** | 303.35 ± 25.47 | 456.80 ± 56.81 | 760.15 ± 61.22 | 185.54 ± 23.63 | 945.69 ± 40.95 |

## B. Why 10B runs out of memory in FP32

The 10B configuration has 12.83 billion parameters in my implementation. Its weights take 47.81 GiB, so the weights alone fit. However, weights, gradients, and AdamW's two moment tensors together need **191.22 GiB** in FP32, before counting activations or temporary tensors. The B200 provides **178.34 GiB**, so this setup cannot fit even with all saved activations freed. The run failed during the first optimizer step, when AdamW allocated its states.

## C. BF16

I reran all sizes with BF16 weights, gradients, and AdamW moments. For 10B, these tensors take 95.61 GiB, and the run completed successfully.

| Model | Forward | Backward | Forward + backward | Optimizer | Total |
|:---|---:|---:|---:|---:|---:|
| **small** | 20.14 ± 4.75 | 41.57 ± 39.73 | 61.71 ± 39.48 | 12.40 ± 0.55 | 74.11 ± 40.00 |
| **medium** | 51.69 ± 48.65 | 67.04 ± 39.86 | 118.73 ± 59.37 | 26.00 ± 1.35 | 144.73 ± 59.88 |
| **large** | 66.47 ± 34.64 | 91.36 ± 41.56 | 157.83 ± 51.58 | 39.48 ± 3.34 | 197.31 ± 52.66 |
| **xl** | 62.17 ± 34.12 | 86.37 ± 37.45 | 148.54 ± 47.64 | 53.19 ± 19.99 | 201.73 ± 63.24 |
| **10B** | 122.33 ± 77.43 | 219.66 ± 37.43 | 341.99 ± 82.50 | 174.34 ± 98.67 | 516.33 ± 149.25 |

## D. BF16 (0 warmup steps)

| Model | Forward | Backward | Forward + backward | Optimizer | Total |
|:---|---:|---:|---:|---:|---:|
| **small** | 91.25 ± 219.15 | 43.45 ± 48.60 | 134.70 ± 267.73 | 14.19 ± 1.13 | 148.88 ± 268.07 |
| **medium** | 115.82 ± 221.54 | 76.23 ± 64.91 | 192.05 ± 286.08 | 25.77 ± 1.90 | 217.82 ± 287.65 |
| **large** | 148.07 ± 241.20 | 97.76 ± 49.78 | 245.82 ± 289.79 | 39.05 ± 3.20 | 284.87 ± 291.96 |
| **xl** | 163.71 ± 302.58 | 91.58 ± 47.13 | 255.29 ± 349.27 | 57.31 ± 36.42 | 312.60 ± 385.24 |
| **10B** | 223.28 ± 293.90 | 229.23 ± 55.23 | 452.51 ± 346.38 | 144.30 ± 8.46 | 596.82 ± 354.53 |

## E. FP32 (0 warmup steps)

| Model | Forward | Backward | Forward + backward | Optimizer | Total |
|:---|---:|---:|---:|---:|---:|
| **small** | 67.36 ± 143.26 | 34.79 ± 37.52 | 102.15 ± 180.77 | 13.24 ± 0.74 | 115.39 ± 180.81 |
| **medium** | 97.68 ± 121.23 | 75.54 ± 28.19 | 173.22 ± 149.14 | 41.78 ± 4.82 | 215.00 ± 144.42 |
| **large** | 231.35 ± 245.08 | 185.73 ± 54.50 | 417.08 ± 277.52 | 67.90 ± 8.92 | 484.98 ± 272.08 |
| **xl** | 379.86 ± 185.85 | 457.67 ± 60.48 | 837.52 ± 241.71 | 192.46 ± 4.21 | 1029.98 ± 245.69 |
