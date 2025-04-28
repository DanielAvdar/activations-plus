"""ReLU and Leaky ReLU variants for PyTorch.

This module provides several simple ReLU-based activation functions.
"""

import torch
from torch import Tensor


def dual_line(x: Tensor, a: float = 1.0, b: float = 0.01, m: float = 0.0) -> Tensor:
    r"""Apply the Dual Line activation.

    .. math::

        \mathrm{DualLine}(x) = \begin{cases} a x + m, & x \geq 0 \\ b x + m, & x < 0 \end{cases}

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Slope for x >= 0 (default 1.0).
    b : float, optional
        Slope for x < 0 (default 0.01).
    m : float, optional
        Offset (default 0.0).

    Returns
    -------
    torch.Tensor
        The element-wise Dual Line activation of the input.


    Source
    ------
    .. seealso::
        A generalized linear activation function discussed in "Survey of Activation Functions for Deep Neural
        Networks" by Nwankpa et al. (2018).

        `arxiv <https://arxiv.org/abs/1811.03378>`_

    Example
    -------


    .. plot:: ../../examples/relu_variants/dual_line_example.py
       :include-source:

    """
    return torch.where(x >= 0, a * x + m, b * x + m)
