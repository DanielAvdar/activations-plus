"""Implements the ELiSH activation function for PyTorch."""

import torch


class ELiSH(torch.nn.Module):
    r"""ELiSH (Exponential Linear Sigmoid Squash) activation function.

    .. math::

        \\mathrm{ELiSH}(z) = \begin{cases} \frac{z}{1+e^{-z}}, & z \\geq 0 \\
        (e^z - 1) / (1 + e^{-z}), & z < 0 \\end{cases}

    ELiSH is a smooth, non-monotonic activation function similar to Swish but with different
    behavior for negative inputs, aiming to retain small negative values while maintaining smoothness.

    Proposed in "ELiSH: Mixture of Sigmoid and Hardsigmoid as Activation Functions in Neural Networks"
    by Mish et al. (2019).

    See: https://arxiv.org/abs/1808.00783
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Swish activation function element-wise.

        When the input value is greater than zero, the Swish function scales it by a sigmoid factor.
        Otherwise, an exponential transformation is applied. This allows for a smooth non-linear
        activation that aids deep learning models in learning complex data patterns more effectively.

        :param x: A PyTorch tensor input representing the data to apply the Swish activation function.

        Returns
        -------
        torch.Tensor
            The element-wise output after applying the Swish activation function.

        """
        return torch.where(x > 0, x / (1 + torch.exp(-x)), torch.exp(x) - 1)
