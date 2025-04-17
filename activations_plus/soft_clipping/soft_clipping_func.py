"""Implements the Soft Clipping activation function for PyTorch."""

from typing import Callable

import torch


class SoftClipping(torch.nn.Module):
    """Soft Clipping activation function.

    This activation function smoothly limits the range of activations, preventing extreme values
    without hard truncation. It is particularly useful for stabilizing neural network training.

    """

    def __init__(self, x_min: float = -1.0, x_max: float = 1.0, clip_func: Callable = torch.sigmoid):
        """Initialize the SoftClipping module.

        SoftClipping is a configurable module for applying soft clipping to values within a specified
        range. The module uses a differentiable non-linear clipping function to constrain input values
        between a defined minimum and maximum range, facilitating gradient-based optimization and
        control over numerical saturation.

        The class allows the customization of the clipping function as well as the range for clipping,
        which makes it applicable in a variety of machine learning tasks requiring such transformations.

        :param x_min: The minimum value to which the input will be clipped. Defaults to -1.0.
        :type x_min: float
        :param x_max: The maximum value to which the input will be clipped. Defaults to 1.0.
        :type x_max: float
        :param clip_func: A differentiable function used to softly clip the input values. This function
            determines the softness and shape of the transition between unclipped and clipped regions.
            Defaults to torch.sigmoid.
        :type clip_func: Callable
        """
        super(SoftClipping, self).__init__()
        self.min_val = x_min
        self.max_val = x_max
        self.clip_func = clip_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a transformation to the input tensor by scaling and clipping its values.

        Scale and clip the input tensor based on predefined `min_val` and `max_val`, using a clipping
        function.

        :param x: Input tensor to be transformed.
        :type x: torch.Tensor
        :return: Transformed tensor after applying the scaling and clipping operation.
        :rtype: torch.Tensor
        """
        return self.min_val + (self.max_val - self.min_val) * self.clip_func(x)
