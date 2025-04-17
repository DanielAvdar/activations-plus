"""Implements the Smooth Rectified Linear Unit (SReLU) activation function for PyTorch."""

# Implementation of the SReLU (S-shaped ReLU), ELiSH (Exponential Linear Sigmoid Squash), and Soft Clipping functions
import torch
from torch import Tensor


class SReLU(torch.nn.Module):
    """SReLU (S-shaped Rectified Linear Unit) activation function."""

    def __init__(self, lower_threshold: float = -1.0, upper_threshold: float = 1.0) -> None:
        """Initialize the SReLU activation function with user-defined thresholds.

        This activation function applies constraints on the input values, where inputs below a specified
        lower threshold or above an upper threshold are clipped. The SReLU is a piecewise linear
        activation function primarily used in neural networks.

        :param lower_threshold: The minimum value an input can take after being passed through the
            activation function.
        :param upper_threshold: The maximum value an input can take after being passed through the
            activation function.
        :raises ValueError: If `lower_threshold` is greater than `upper_threshold`.
        """
        super(SReLU, self).__init__()
        if lower_threshold > upper_threshold:
            raise ValueError("lower_threshold must be less than or equal to upper_threshold")
        self.lower_threshold: float = lower_threshold
        self.upper_threshold: float = upper_threshold

    def forward(self, x: Tensor) -> Tensor:
        """Apply a thresholding operation to the input tensor, clipping values that are below or above.

        the specified thresholds.

        If a value in the input tensor is less than the `lower_threshold`, it is replaced with
        `lower_threshold`. If a value exceeds the `upper_threshold`, it is replaced with
        `upper_threshold`. Values in between these thresholds remain unchanged.

        :param x: Input tensor to which the thresholding operation is applied.
        :type x: Tensor
        :return: A tensor with values clipped according to the thresholding criteria.
        :rtype: Tensor
        """
        return torch.where(
            x < self.lower_threshold,
            self.lower_threshold,
            torch.where(x > self.upper_threshold, self.upper_threshold, x),
        )
