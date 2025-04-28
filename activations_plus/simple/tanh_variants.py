"""Tanh-based activation functions and their variants for neural networks."""

import torch
from torch import Tensor


def penalized_tanh(x: Tensor, a: float = 0.25) -> Tensor:
    r"""Apply the Penalized Tanh activation function.

    .. math::

        \text{PenalizedTanh}(x) = \tanh(x) - a \cdot \tanh^2(x)



    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Penalty coefficient (default 0.25).

    Returns
    -------
    torch.Tensor
        The element-wise Penalized Tanh of the input.



    Source
    ------
    .. seealso::
        A variant of tanh activation function proposed in "Activation Functions: Comparison in Neural
        Network Architecture" by Sharma et al. (2021).

        `arxiv <https://arxiv.org/abs/2109.14545>`_

    Example
    -------
    .. plot:: ../../examples/tanh_variants/penalized_tanh_example.py
       :include-source:

    """
    t = torch.tanh(x)
    return t - a * t**2
