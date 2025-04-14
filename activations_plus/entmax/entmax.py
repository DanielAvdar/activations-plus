import torch.nn as nn

from .entmax_func import Entmax15Function


class Entmax(nn.Module):
    __constants__ = ["dim"]

    def __init__(self, dim=-1):
        """
        Entmax15 activation with α=1.5 from https://arxiv.org/abs/1905.05702
        Parameters
        ----------
        dim: The dimension to apply the activation.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return Entmax15Function.apply(x, self.dim)

    def extra_repr(self):
        return f"dim={self.dim}"
