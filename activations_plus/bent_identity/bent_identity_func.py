"""Implements the Bent Identity activation function for PyTorch."""

import torch


class BentIdentity(torch.nn.Module):
    """Bent Identity activation function.

    This activation function provides a smooth approximation of the identity function.
    It introduces non-linearity while preserving the identity mapping for large inputs.

    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a custom transformation of the input tensor.

        Perform a combination of squaring, square rooting, and addition. The result is adjusted and
        normalized using specific constants.

        :param x: A tensor of numeric values on which the custom transformation is performed. The tensor
            should consist of real-valued numbers.
        :type x: torch.Tensor
        :return: A tensor containing the transformed values after applying the custom operation to the
            input tensor.
        :rtype: torch.Tensor
        """
        return (torch.sqrt(x**2 + 1) - 1) / 2 + x
