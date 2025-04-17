"""Implements the ELiSH activation function for PyTorch."""

import torch


class ELiSH(torch.nn.Module):
    """ELiSH (Exponential Linear Sigmoid Squash) activation function.

    Combines properties of exponential and sigmoid functions,
     aiming to retain small negative values while maintaining smoothness.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Swish activation function element-wise.

        When the input value is greater than zero, the Swish function scales it by a sigmoid factor.
        Otherwise, an exponential transformation is applied. This allows for a smooth non-linear
        activation that aids deep learning models in learning complex data patterns more effectively.

        :param x: A PyTorch tensor input representing the data to apply the Swish activation function.
        :return: A PyTorch tensor containing the element-wise output after applying the Swish activation
            function.
        """
        return torch.where(x > 0, x / (1 + torch.exp(-x)), torch.exp(x) - 1)
