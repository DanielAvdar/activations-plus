"""Implements the Sparsemax activation function for PyTorch."""

import torch.nn as nn
from torch import Tensor

from .sparsemax_func import SparsemaxFunction


class Sparsemax(nn.Module):
    """Sparsemax class implements a transformation function from the paper.

    "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label
    Classification" (https://arxiv.org/pdf/1602.02068.pdf). This function is
    used as an activation function that is similar to softmax but can produce
    sparse output, where some of the entries are exactly zero.

    This class is designed to handle the computation over a specified
    dimension, and it can be used as a module in neural network architectures.

    """

    __constants__ = ["dim"]

    def __init__(self, dim: int = -1) -> None:
        """Initialize the Sparsemax activation function.

        Parameters
        ----------
        dim : int, optional
            The dimension along which to apply the Sparsemax operation. Defaults to -1, indicating the last dimension.

        """
        super(Sparsemax, self).__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """Apply the sparsemax function along the specified dimension.

        Sparsemax is a neural network activation function that maps input logits to probabilities,
        similar to softmax. Unlike softmax, it can lead to sparse probability distributions where some
        probabilities are exactly zero.

        :param x: The input tensor to which the sparsemax function will be applied.
        :return: The tensor after applying the sparsemax operation along the specified dimension.
        """
        return SparsemaxFunction.apply(x, self.dim)
