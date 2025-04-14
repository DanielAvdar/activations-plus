import torch.nn as nn

from .sparsemax_func import SparsemaxFunction


class Sparsemax(nn.Module):
    __constants__ = ["dim"]

    def __init__(self, dim=-1):
        """
        Sparsemax class as seen in https://arxiv.org/pdf/1602.02068.pdf
        Parameters
        ----------
        dim: The dimension we want to cast the operation over. Default -1
        """
        super(Sparsemax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "dim"):
            self.dim = None

    def forward(self, x):
        return SparsemaxFunction.apply(x, self.dim)

    def extra_repr(self):
        return f"dim={self.dim}"
