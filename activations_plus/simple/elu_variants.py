"""ELU-based activation functions and their variants for neural networks."""

import torch
from torch import Tensor


def isrlu(x: Tensor, alpha: float = 1.0) -> Tensor:
    r"""Apply the Inverse Square Root Linear Unit activation function.

    .. math::

        \text{ISRLU}(x) = \begin{cases}
            x, & x \geq 0 \\
            \frac{x}{\sqrt{1 + \alpha x^2}}, & x < 0
        \end{cases}


    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    alpha : float, optional
        Scaling parameter (default 1.0).

    Returns
    -------
    torch.Tensor
        The element-wise ISRLU of the input.



    Source
    ------
    .. seealso::
        Proposed in **"Improving Deep Neural Networks with New Activation Functions"**
        by Carlile et al. (2017).

        `arxiv <https://arxiv.org/abs/1710.09967>`_

    Example
    -------

    .. plot:: ../../examples/elu_variants/isrlu_example.py
       :include-source:

    """
    return torch.where(x >= 0, x, x / torch.sqrt(1 + alpha * x.pow(2)))


def pelu(x: Tensor, alpha: float = 1.0, beta: float = 1.0) -> Tensor:
    r"""Parametric Exponential Linear Unit (PELU) activation function.

    .. math::
        \text{PELU}(x) = \begin{cases}
            \frac{\alpha}{\beta} \times x, & \text{if } x \geq 0 \\
            \alpha \times (e^{x/\beta} - 1), & \text{if } x < 0
        \end{cases}

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    alpha : float, optional
        The alpha parameter for scaling (default 1.0).
    beta : float, optional
        The beta parameter for controlling the negative part (default 1.0).

    Returns
    -------
    torch.Tensor
        Output tensor with the PELU activation applied.


    Source
    ------
    .. seealso::
        Proposed in **"Parametric exponential linear unit for deep convolutional neural networks"**
        by Trottier et al. (2016).

        `arxiv <https://arxiv.org/abs/1605.09332>`_

    Example
    -------

    .. plot:: ../../examples/elu_variants/pelu_example.py
       :include-source:

    References
    ----------
    .. [1] Trottier, L., Gigu, P., Chaib-draa, B., & Bengio, Y. (2016).
           Parametric exponential linear unit for deep convolutional neural networks.
           arXiv preprint arXiv:1605.09332.

    """
    pos_part = (alpha / beta) * torch.clamp(x, min=0)
    neg_part = alpha * (torch.exp(torch.clamp(x, max=0) / beta) - 1)
    return pos_part + neg_part
