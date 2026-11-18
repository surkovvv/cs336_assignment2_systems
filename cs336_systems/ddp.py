import torch
import torch.nn as nn
import torch.distributed as dist


class NaiveDDP(nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

        for _, param in self.module.named_parameters():
            with torch.no_grad():
                dist.broadcast(param, src=0)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.module(data)

    def finish_gradient_synchronization(self, optimizer: torch.optim.Optimizer):
        for _, param in self.module.named_parameters():
            if param.requires_grad:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
