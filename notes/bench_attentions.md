### Without compilation

| `seq_len` \ `d_model` | 16 | 32 | 64 | 128 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | Forward: 0.358 ms<br>Backward: 0.688 ms<br>Memory: 69.02 MiB | Forward: 0.343 ms<br>Backward: 0.633 ms<br>Memory: 70.02 MiB | Forward: 0.338 ms<br>Backward: 0.656 ms<br>Memory: 72.02 MiB | Forward: 0.342 ms<br>Backward: 0.669 ms<br>Memory: 76.02 MiB |
| 1024 | Forward: 0.350 ms<br>Backward: 0.770 ms<br>Memory: 132.09 MiB | Forward: 0.358 ms<br>Backward: 0.759 ms<br>Memory: 136.09 MiB | Forward: 0.360 ms<br>Backward: 0.768 ms<br>Memory: 144.09 MiB | Forward: 0.378 ms<br>Backward: 0.803 ms<br>Memory: 160.09 MiB |
| 4096 | Forward: 3.007 ms<br>Backward: 6.631 ms<br>Memory: 1104.38 MiB | Forward: 3.062 ms<br>Backward: 6.674 ms<br>Memory: 1120.38 MiB | Forward: 3.349 ms<br>Backward: 7.222 ms<br>Memory: 1152.38 MiB | Forward: 3.951 ms<br>Backward: 8.434 ms<br>Memory: 1216.38 MiB |
| 8192 | Forward: 11.529 ms<br>Backward: 25.468 ms<br>Memory: 4192.75 MiB | Forward: 11.868 ms<br>Backward: 25.815 ms<br>Memory: 4224.75 MiB | Forward: 13.036 ms<br>Backward: 28.133 ms<br>Memory: 4288.75 MiB | Forward: 15.359 ms<br>Backward: 32.582 ms<br>Memory: 4416.75 MiB |
| 16384 | Forward: 45.159 ms<br>Backward: 100.670 ms<br>Memory: 16513.50 MiB | Forward: 46.473 ms<br>Backward: 102.064 ms<br>Memory: 16577.50 MiB | Forward: 51.581 ms<br>Backward: 111.576 ms<br>Memory: 16705.50 MiB | Forward: 60.543 ms<br>Backward: 129.201 ms<br>Memory: 16961.50 MiB |

### With torch.compile

| `seq_len` \ `d_model` | 16 | 32 | 64 | 128 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | Forward: 0.214 ms (−40.35%)<br>Backward: 0.348 ms (−49.43%)<br>Memory: 69.03 MiB (+0.011%) | Forward: 0.243 ms (−29.15%)<br>Backward: 0.352 ms (−44.39%)<br>Memory: 70.03 MiB (+0.011%) | Forward: 0.239 ms (−29.21%)<br>Backward: 0.345 ms (−47.37%)<br>Memory: 72.03 MiB (+0.011%) | Forward: 0.237 ms (−30.83%)<br>Backward: 0.340 ms (−49.22%)<br>Memory: 76.03 MiB (+0.010%) |
| 1024 | Forward: 0.251 ms (−28.29%)<br>Backward: 0.421 ms (−45.28%)<br>Memory: 132.12 MiB (+0.024%) | Forward: 0.245 ms (−31.35%)<br>Backward: 0.451 ms (−40.60%)<br>Memory: 136.12 MiB (+0.023%) | Forward: 0.352 ms (−2.18%)<br>Backward: 0.476 ms (−38.08%)<br>Memory: 144.12 MiB (+0.022%) | Forward: 0.385 ms (+1.93%)<br>Backward: 0.561 ms (−30.19%)<br>Memory: 160.12 MiB (+0.020%) |
| 4096 | Forward: 1.190 ms (−60.43%)<br>Backward: 2.677 ms (−59.63%)<br>Memory: 1104.50 MiB (+0.011%) | Forward: 1.707 ms (−44.26%)<br>Backward: 3.052 ms (−54.27%)<br>Memory: 1120.50 MiB (+0.011%) | Forward: 1.786 ms (−46.65%)<br>Backward: 3.272 ms (−54.70%)<br>Memory: 1152.50 MiB (+0.011%) | Forward: 2.404 ms (−39.16%)<br>Backward: 4.530 ms (−46.29%)<br>Memory: 1216.50 MiB (+0.010%) |
| 8192 | Forward: 4.966 ms (−56.93%)<br>Backward: 10.521 ms (−58.69%)<br>Memory: 4193.00 MiB (+0.006%) | Forward: 6.374 ms (−46.29%)<br>Backward: 11.596 ms (−55.08%)<br>Memory: 4225.00 MiB (+0.006%) | Forward: 6.096 ms (−53.24%)<br>Backward: 12.585 ms (−55.26%)<br>Memory: 4289.00 MiB (+0.006%) | Forward: 8.521 ms (−44.52%)<br>Backward: 17.162 ms (−47.33%)<br>Memory: 4417.00 MiB (+0.006%) |
| 16384 | Forward: 16.302 ms (−63.90%)<br>Backward: 38.390 ms (−61.87%)<br>Memory: 16514.00 MiB (+0.003%) | Forward: 18.278 ms (−60.67%)<br>Backward: 39.837 ms (−60.97%)<br>Memory: 16578.00 MiB (+0.003%) | Forward: 23.847 ms (−53.77%)<br>Backward: 50.063 ms (−55.13%)<br>Memory: 16706.00 MiB (+0.003%) | Forward: 32.842 ms (−45.75%)<br>Backward: 67.931 ms (−47.42%)<br>Memory: 16962.00 MiB (+0.003%) |

![Attention forward: without compilation on the left, with torch.compile on the right; matching axis scales.](pictures/attention_forward_comparison.png)

![Attention backward: without compilation on the left, with torch.compile on the right; matching axis scales.](pictures/attention_backward_comparison.png)

![Attention memory: without compilation on the left, with torch.compile on the right; matching axis scales.](pictures/attention_memory_comparison.png)

