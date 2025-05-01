"""ReLU and Leaky ReLU variants for PyTorch.

This module provides several simple ReLU-based activation functions.
"""

import torch
from torch import Tensor


def dual_line(x: Tensor, a: float = 0.5, b: float = 0.5) -> Tensor:
    r"""Apply the Dual Line activation function.

    .. math::

        \text{DualLine}(x) = \begin{cases}
            a \cdot x, & \text{if } x < 0 \\
            b \cdot x, & \text{if } x \geq 0
        \end{cases}

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Negative slope coefficient (default 0.5).
    b : float, optional
        Positive slope coefficient (default 0.5).

    Returns
    -------
    torch.Tensor
        The element-wise Dual Line of the input.


    Source
    ------
    .. seealso::
        A generalized linear activation function discussed in **"Survey of Activation Functions for Deep Neural
        Networks"** by Nwankpa et al. (2018).

        `arxiv <https://arxiv.org/abs/1811.03378>`_

    Example
    -------


    .. plot:: ../../examples/relu_variants/dual_line_example.py
       :include-source:

    """
    return torch.where(x < 0, a * x, b * x)
