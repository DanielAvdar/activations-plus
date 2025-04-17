"""Module implementing the Maxout activation function for PyTorch.

This module provides a Maxout activation class that can be used in neural network architectures
for learning piecewise linear convex functions.
"""

import torch


class Maxout(torch.nn.Module):
    """Maxout activation function.

    Select the maximum across multiple linear functions,
    allowing the network to learn piecewise linear convex functions.

    """

    def __init__(self, num_pieces: int) -> None:
        """Initialize the Maxout activation module.

        Initialize the number of pieces into which the input is split for the Maxout operation.

        :param num_pieces: Number of pieces into which the input is divided for the
            Maxout operation.
        """
        super(Maxout, self).__init__()
        self.num_pieces = num_pieces

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape the input tensor and compute the maximum along the split dimension.

        Reshape the input tensor so that the last dimension is divided into `num_pieces`,
        then compute and return the maximum values along the new axis.

        :param x: A tensor of arbitrary shape where the last dimension must be divisible by
            `self.num_pieces`.
        :return: A tensor containing the maximum values along the split dimension of the
            reshaped input tensor. The resulting shape will match all but the last dimension
            of the input tensor.
        """
        shape = x.shape[:-1] + (x.shape[-1] // self.num_pieces, self.num_pieces)
        x = x.view(*shape)
        return x.max(-1)[0]
