from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

class LinearTorch(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.weight = weight.T
        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)
    
class Linear(nn.Module):
    def __init__(
        self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #print(f"===== {input.shape=}, {input.is_contiguous()=}, {self.weight.is_contiguous()=}, {self.bias is None}")
        #return F.linear(input, self.weight, self.bias)
        return torch.matmul(input, self.weight)
