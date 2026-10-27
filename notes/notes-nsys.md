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

### (b) GPU kernel summary

FP32, batch size 4. Instances and total GPU times are summed over 10 measured iterations. Each row shows the kernel with the highest cumulative GPU time in that pass. Forward + backward includes loss computation.

**Answer.** I summed the GPU kernel times over 10 forward passes and 10 forward + backward passes, with the leading kernels and their invocation counts shown below for each configuration. The leading kernel does not always stay the same: its full name matches for `xl_seq128`, while `small_seq128` changes from the `tnn` variant to `nnn`.

#### small_seq128

| Pass | Kernel name | Instances | Total GPU time (ms) | GPU time (%) |
|:---|:---|---:|---:|---:|
| Forward | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x32x16_1x1x1_3_tnn_align1_bias_f32_relu` | 600 | 22.116193 | — |
| Forward + backward | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x32x16_1x1x1_3_nnn_align1_bias_f32_relu` | 960 | 31.993063 | 18.8 |

#### small_seq512

| Pass | Kernel name | Instances | Total GPU time (ms) | GPU time (%) |
|:---|:---|---:|---:|---:|
| Forward | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x128x16_1x1x1_3_tnn_align1_bias_f32_relu` | 250 | 53.699046 | — |
| Forward + backward | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_nnn_align1_bias_f32_relu` | 970 | 110.578643 | 23.4 |

#### small_seq4096

| Pass | Kernel name | Instances | Total GPU time (ms) | GPU time (%) |
|:---|:---|---:|---:|---:|
| Forward | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x128x16_1x1x1_3_tnn_align1_bias_f32_relu` | 250 | 359.921876 | — |
| Forward + backward | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 1960 | 1257.656005 | 15.9 |

#### xl_seq128

| Pass | Kernel name | Instances | Total GPU time (ms) | GPU time (%) |
|:---|:---|---:|---:|---:|
| Forward | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x32x16_1x1x1_3_tnn_align1_bias_f32_relu` | 1600 | 469.669784 | — |
| Forward + backward | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x32x16_1x1x1_3_tnn_align1_bias_f32_relu` | 1600 | 468.422099 | 17.3 |

#### xl_seq512

| Pass | Kernel name | Instances | Total GPU time (ms) | GPU time (%) |
|:---|:---|---:|---:|---:|
| Forward | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x128x16_1x1x1_3_tnn_align1_bias_f32_relu` | 650 | 1245.339287 | — |
| Forward + backward | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_nnn_align1_bias_f32_relu` | 1930 | 1867.836112 | 21.5 |

#### xl_seq1024

| Pass | Kernel name | Instances | Total GPU time (ms) | GPU time (%) |
|:---|:---|---:|---:|---:|
| Forward | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x64x16_1x1x1_3_tnn_align1_bias_f32_relu` | 1600 | 2447.365485 | — |
| Forward + backward | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x64x16_1x1x1_3_nnn_align1_bias_f32_relu` | 1930 | 3435.970532 | 19.6 |

The forward summaries did not include GPU time percentages (—).

### (c) Other kernels in the forward pass

**Answer.** The forward profiles include elementwise multiplication, division and addition, as well as a vectorized multiplication kernel. The selected kernel names, invocation counts and total GPU times are listed below.

FP32, batch size 4. These are selected rows from the forward top-5 results in the notebook. Instances and total GPU times are summed over 10 measured forward passes.

#### small_seq128

| Kernel name | Operation | Instances | Total GPU time (ms) |
|:---|:---|---:|---:|
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | Elementwise multiplication | 1210 | 3.245927 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | Copy | 960 | 2.820553 |

#### small_seq512

| Kernel name | Operation | Instances | Total GPU time (ms) |
|:---|:---|---:|---:|
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | Elementwise division | 370 | 5.999146 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | Elementwise multiplication | 1210 | 5.701963 |

#### small_seq4096

| Kernel name | Operation | Instances | Total GPU time (ms) |
|:---|:---|---:|---:|
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | Elementwise division | 490 | 320.358167 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::<unnamed>::masked_fill_kernel(at::TensorIterator &, const c10::Scalar &)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float, bool) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | Masked fill | 240 | 278.225075 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | Elementwise addition | 240 | 268.981014 |

#### xl_seq128

| Kernel name | Operation | Instances | Total GPU time (ms) |
|:---|:---|---:|---:|
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | Elementwise multiplication | 3210 | 13.810398 |
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | Copy | 2560 | 12.521655 |
| `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | Elementwise multiplication (vectorized) | 1290 | 6.500979 |

#### xl_seq1024

| Kernel name | Operation | Instances | Total GPU time (ms) |
|:---|:---|---:|---:|
| `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | Elementwise division | 970 | 158.272109 |

### (d) Forward and full training-step statistics

FP32, batch size 4, 10 measured iterations. F is forward only (regime 0); FBO is forward + loss computation + backward + optimizer (regime 2). All runs use Capture API to exclude warmup.

Matrix multiplication includes all kernel names containing `sgemm`, including both `cutlass3x` and `cutlass_80` kernels. Percentages are calculated from the unrounded `Total Time (ns)` values, using the sum of CUDA kernel durations in each captured run as the denominator, and displayed to one decimal place.

**Grad-mode note.** The small F run uses `torch.no_grad()`. The XL F run predates that change and has autograd enabled; it still contains forward passes only.

#### Matrix-multiplication share of GPU kernel time (%)

| Pass | small_seq128 | small_seq512 | small_seq4096 | xl_seq128 | xl_seq512 | xl_seq1024 |
|:---|---:|---:|---:|---:|---:|---:|
| F | 70.2 | 73.7 | 45.7 | 92.5 | 89.8 | 85.2 |
| FBO | 50.4 | 64.0 | 45.1 | 68.6 | 81.2 | 81.0 |

#### Top 3 non-GEMM kernels by cumulative GPU time

##### small_seq128

| Pass | Kernel name | GPU time (%) |
|:---|:---|---:|
| F | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 5.6 |
| F | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 4.9 |
| F | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 2.6 |
| FBO | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)2>>(int, T2, T3)` | 9.7 |
| FBO | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 8.7 |
| FBO | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 7.2 |

##### small_seq512

| Pass | Kernel name | GPU time (%) |
|:---|:---|---:|
| F | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 3.8 |
| F | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 3.6 |
| F | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 3.3 |
| FBO | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 5.0 |
| FBO | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 4.7 |
| FBO | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 4.3 |

##### small_seq4096

| Pass | Kernel name | GPU time (%) |
|:---|:---|---:|
| F | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 11.9 |
| F | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::<unnamed>::masked_fill_kernel(at::TensorIterator &, const c10::Scalar &)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float, bool) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 10.4 |
| F | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 10.0 |
| FBO | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 15.7 |
| FBO | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::<unnamed>::masked_fill_kernel(at::TensorIterator &, const c10::Scalar &)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float, bool) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 7.0 |
| FBO | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 6.8 |

##### xl_seq128

| Pass | Kernel name | GPU time (%) |
|:---|:---|---:|
| F | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 1.5 |
| F | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 1.3 |
| F | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 0.7 |
| FBO | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)2>>(int, T2, T3)` | 7.8 |
| FBO | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 7.6 |
| FBO | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 5.8 |

##### xl_seq512

| Pass | Kernel name | GPU time (%) |
|:---|:---|---:|
| F | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 1.5 |
| F | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 1.2 |
| F | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 1.1 |
| FBO | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 3.3 |
| FBO | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)2>>(int, T2, T3)` | 2.8 |
| FBO | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 2.8 |

##### xl_seq1024

| Pass | Kernel name | GPU time (%) |
|:---|:---|---:|
| F | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 2.7 |
| F | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 2.1 |
| F | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::<unnamed>::masked_fill_kernel(at::TensorIterator &, const c10::Scalar &)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float, bool) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 2.1 |
| FBO | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 3.4 |
| FBO | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)` | 2.3 |
| FBO | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)` | 1.9 |

#### Source runs

- F, small: [20261021T200854Z-nsys-fp32-b671f46a](</Users/tr3n1ttty/code projects/preps/cs 336/assignment2-systems/benches/nsys_runs/20261021T200854Z-nsys-fp32-b671f46a/metadata.json>).
- F, xl: [20261021T195859Z-nsys-fp32-e2fa6c8f](</Users/tr3n1ttty/code projects/preps/cs 336/assignment2-systems/benches/nsys_runs/20261021T195859Z-nsys-fp32-e2fa6c8f/metadata.json>).
- FBO, small: [20261021T192723Z-nsys-fp32-80973f26](</Users/tr3n1ttty/code projects/preps/cs 336/assignment2-systems/benches/nsys_runs/20261021T192723Z-nsys-fp32-80973f26/metadata.json>).
- FBO, xl: [20261021T192722Z-nsys-fp32-e29c609f](</Users/tr3n1ttty/code projects/preps/cs 336/assignment2-systems/benches/nsys_runs/20261021T192722Z-nsys-fp32-e29c609f/metadata.json>).
