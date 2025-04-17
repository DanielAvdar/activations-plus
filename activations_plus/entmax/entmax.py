"""Implements the Entmax activation function for PyTorch."""

import torch.nn as nn
from torch import Tensor

from .entmax_func import Entmax15Function


class Entmax(nn.Module):
    """A neural network module implementing the Entmax15 activation function with α=1.5.

    This activation function is based on the paper "Sparse Transformers: Sparsity-preserving
    activations" (https://arxiv.org/abs/1905.05702). It provides a sparse probability distribution
    over inputs, making it suitable for attention mechanisms and tasks requiring sparsity.

    """

    __constants__ = ["dim"]

    def __init__(self, dim: int = -1) -> None:
        """Entmax15 activation with α=1.5 from https://arxiv.org/abs/1905.05702.

        Parameters.
        ----------
        dim: The dimension to apply the activation.

        """
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """Apply the Entmax15 function along a specified dimension.

        Entmax15 is a smooth variation of softmax that includes the capability to sparsify the output.
        It is commonly used in machine learning tasks such as natural language processing where sparse,
        non-negative distributions are desired.

        :param x: The input tensor on which the Entmax15 function will be applied.
        :return: The tensor obtained after applying the Entmax15 transformation to the input tensor. The
            output tensor has the same shape as the input but may exhibit sparse behavior depending on the
            input values.
        """
        return Entmax15Function.apply(x, self.dim)
