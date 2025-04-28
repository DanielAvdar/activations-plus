"""Sigmoid-based activation functions and their variants for neural networks."""

import torch
from torch import Tensor


def new_sigmoid(x: Tensor, a: float = 0.5) -> Tensor:
    r"""Apply the New Sigmoid activation function.

    .. math::

        \text{NewSigmoid}(z) = \begin{cases}
            \frac{z}{a(1 + |z|) + 1}, & z < 0, \\
            \frac{z}{(1 + |z|) + a}, & z \geq 0,
        \end{cases}



    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Shape parameter (default 0.5).

    Returns
    -------
    torch.Tensor
        The element-wise New Sigmoid of the input.


    Source
    ------
    .. seealso::
        Proposed in "New Sigmoid-Like Activation Function for Neural Networks" by Hu et al. (2023).

        `arxiv <https://arxiv.org/abs/2302.01931>`_


    Example
    -------
    .. plot:: ../../examples/sigmoid_variants/new_sigmoid_example.py
       :include-source:

    """
    ax = torch.abs(x)
    return torch.where(x < 0, x / (a * (1 + ax) + 1), x / ((1 + ax) + a))


def root2sigmoid(x: Tensor) -> Tensor:
    r"""Apply the Root-2 Sigmoid activation function.

    .. math::

        \text{Root2Sigmoid}(z) = \frac{z}{\sqrt{1 + z^2}}



    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Root-2 Sigmoid of the input.


    Source
    ------
    .. seealso::

        A root-based variation of the sigmoid function, analyzed in "An Alternative
        to the Sigmoid Function" by Schert et al. (2020).

        `arxiv <https://arxiv.org/abs/2008.07861>`_

    Example
    -------
    .. plot:: ../../examples/sigmoid_variants/root2sigmoid_example.py
       :include-source:

    """
    return x / torch.sqrt(1 + x**2)


def isrlu(x: Tensor, alpha: float = 1.0) -> Tensor:
    r"""Apply the Inverse Square Root Linear Unit activation function.

    .. math::

        \text{ISRLU}(z) = \begin{cases}
            \frac{z}{\sqrt{1 + \alpha z^2}}, & z < 0 \\
            z, & z \geq 0
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
        The element-wise ISRLU activation of the input.


    Source
    ------
    .. seealso::
        Proposed in "Improving Deep Neural Networks with New Activation Functions"
        by Carlile et al. (2017).

        `arxiv <https://arxiv.org/abs/1710.09967>`_

    Example
    -------

    .. plot:: ../../examples/sigmoid_variants/isrlu_example.py
       :include-source:

    """
    return torch.where(x >= 0, x, x / torch.sqrt(1 + alpha * x**2))


def sigmoid_gumbel(x: Tensor, alpha: float = 1.0) -> Tensor:
    r"""Apply the Sigmoid-Gumbel activation function.

    .. math::

        \text{SigmoidGumbel}(z) = \frac{1}{1 + \exp(-\alpha z - \exp(-\alpha z))}



    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    alpha : float, optional
        Scaling parameter (default 1.0).

    Returns
    -------
    torch.Tensor
        The element-wise Sigmoid-Gumbel activation of the input.


    Source
    ------
    .. seealso::

        Developed based on both sigmoid and Gumbel distributions in
        "Novel Activation Functions for Enhanced Neural Network Performance"
        by Kumar et al. (2020).

        `arxiv <https://arxiv.org/abs/2012.07431>`_

    Example
    -------


    .. plot:: ../../examples/sigmoid_variants/sigmoid_gumbel_example.py
       :include-source:

    """
    temp = -alpha * x - torch.exp(-alpha * x)
    return 1 / (1 + torch.exp(temp))
