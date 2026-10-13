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