"""Implements the Sparsemax activation function for PyTorch."""

import torch
from torch import nn

from activations_plus.sparsemax.sparsemax_func_v2 import sparsemax


class Sparsemax(nn.Module):
    def __init__(self, dim: int = -1):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return sparsemax(input_, self.dim)
