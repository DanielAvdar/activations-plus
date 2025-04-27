"""Sigmoid-based activation functions and their variants for neural networks."""

import torch
from torch import Tensor


def rootsig(x: Tensor, a: float = 1.0) -> Tensor:
    r"""Apply the Rootsig activation function (also called Unnamed Sigmoid 3).

    .. math::

        \text{Rootsig}(z) = \frac{az}{\sqrt{1 + a^2z^2}}

    Proposed in "An Extensive Study of Activation Functions in Deep Neural Networks"
    by Dubey et al. (2022).

    See: https://arxiv.org/abs/2202.00442

    .. plot:: ../../examples/sigmoid_variants/rootsig_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Scale parameter (default 1.0).

    Returns
    -------
    torch.Tensor
        The element-wise Rootsig of the input.

    """
    return (a * x) / torch.sqrt(1 + (a * x) ** 2)


def new_sigmoid(x: Tensor) -> Tensor:
    r"""Apply the New Sigmoid activation function.

    .. math::

        \text{NewSigmoid}(z) = \frac{\exp(z) - \exp(-z)}{2(\exp(2z) + \exp(-2z))}

    Introduced in "New Activation Functions for Complex-Valued Neural Network"
    by Aizenberg et al. (2011).

    See: https://arxiv.org/abs/1202.2676

    .. plot:: ../../examples/sigmoid_variants/new_sigmoid_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise New Sigmoid of the input.

    """
    return (torch.exp(x) - torch.exp(-x)) / (2 * (torch.exp(2 * x) + torch.exp(-2 * x)))


def sigmoid_gumbel(x: Tensor) -> Tensor:
    r"""Apply the Sigmoid Gumbel activation function.

    .. math::

        \text{SigmoidGumbel}(z) = \frac{1}{1 + \exp(-z) \exp(-\exp(-z))}

    Based on the Gumbel distribution, described in "Novel Activation Functions for Neural Networks
    using the Gumbel Statistical Distribution" by Martin et al. (2019).

    See: https://arxiv.org/abs/1908.01000

    .. plot:: ../../examples/sigmoid_variants/sigmoid_gumbel_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Sigmoid Gumbel of the input.

    """
    return 1 / (1 + torch.exp(-x) * torch.exp(-torch.exp(-x)))


def root2sigmoid(x: Tensor) -> Tensor:
    r"""Apply the Root2sigmoid activation function.

    .. math::

        \text{Root2sigmoid}(z) = \frac{\sqrt{2}z}{\sqrt{2^{-2z}} + \sqrt{2^{2z}}}

    Proposed in "Comprehensive Analysis of Different Activation Functions in Deep Learning"
    by Kumar et al. (2021).

    See: https://arxiv.org/abs/2101.09957

    .. plot:: ../../examples/sigmoid_variants/root2sigmoid_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Root2sigmoid of the input.

    """
    sqrt2 = torch.sqrt(torch.tensor(2.0))
    return (sqrt2 * x) / (torch.sqrt(torch.pow(2, -2 * x)) + torch.sqrt(torch.pow(2, 2 * x)))
