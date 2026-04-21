"""ConforNet for conditioning Pairformer representations."""

import torch
import torch.nn as nn


class ConforNet(nn.Module):
    """Learnable affine transformation: x @ W^T + b. Initialized to identity."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.W = nn.Parameter(torch.eye(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.W.t()) + self.b

    def reset_to_identity(self):
        with torch.no_grad():
            self.W.copy_(torch.eye(self.dim, device=self.W.device, dtype=self.W.dtype))
            self.b.zero_()
