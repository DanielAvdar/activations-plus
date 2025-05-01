"""Implementations of GELU, Swish and related activation functions."""

import torch
from torch import Tensor


def swish(x: Tensor, beta: float = 1.0) -> Tensor:
    r"""Swish activation function.

    .. math::
        \text{Swish}(x) = x \cdot \sigma(\beta x)

    where :math:`\sigma(z)` is the sigmoid function and :math:`\beta` is a parameter.

    When beta=1.0, this is equivalent to the SiLU (Sigmoid Linear Unit) function.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    beta : float, optional
        Scaling parameter for the sigmoid (default 1.0).

    Returns
    -------
    torch.Tensor
        Output tensor with the Swish activation applied.


    Source
    ------
    .. seealso::
        Proposed in **"Searching for activation functions"**
        by Ramachandran et al. (2017).

        `arxiv <https://arxiv.org/abs/1710.05941>`_

    Example
    -------

    .. plot:: ../../examples/gelu_swish_variants/swish_example.py
       :include-source:

    References
    ----------
    .. [1] Ramachandran, P., Zoph, B., & Le, Q. V. (2017). Searching for activation functions.
           arXiv preprint arXiv:1710.05941.

    """
    return x * torch.sigmoid(beta * x)
