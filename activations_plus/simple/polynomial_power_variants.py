"""Polynomial and power-based activations for PyTorch.

This module provides several simple polynomial and power-based activation functions.
"""

import torch
from torch import Tensor


def polynomial_linear_unit(x: Tensor, alpha: float = 0.1, c: float = 1.0) -> Tensor:
    r"""Apply the Polynomial Linear Unit activation function.

    .. math::

        \text{PoLU}(z) = \begin{cases}
            z, & z \geq 0, \\
            z + \alpha z^c, & z < 0,
        \end{cases}

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    alpha : float, optional
        Coefficient for the polynomial term (default 0.1).
    c : float, optional
        Power for the polynomial term (default 1.0).

    Returns
    -------
    torch.Tensor
        The element-wise PoLU of the input.


    Source
    ------
    .. seealso::
        Proposed in "PoLU: A Learnable Activation Function with Explicit Noise-Robust Characteristics"
        by Liu et al. (2021).

        `arxiv <https://arxiv.org/abs/2110.12911>`_

    Example
    -------


    .. plot:: ../../examples/polynomial_power_variants/polynomial_linear_unit_example.py
       :include-source:

    """
    return torch.where(x >= 0, x, x + alpha * torch.pow(torch.abs(x), c))


def power_function_linear_unit(x: Tensor, alpha: float = 1.0, beta: float = 1.0) -> Tensor:
    r"""Apply the Power Function Linear Unit activation function.

    .. math::

        \text{PFLU}(z) = \begin{cases}
            z, & z \geq 0, \\
            -\alpha \left(1 - \left(1 + \frac{z}{\beta}\right)^{\beta}\right), & z < 0,
        \end{cases}

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    alpha : float, optional
        Scale parameter (default 1.0).
    beta : float, optional
        Shape parameter (default 1.0).

    Returns
    -------
    torch.Tensor
        The element-wise PFLU of the input.


    Source
    ------
    .. seealso::
        Introduced in "Power Function Linear Units (PFLU) for Improving Deep Network Training"
        by Li et al. (2022).

        `arxiv <https://arxiv.org/abs/2208.08408>`_

    Example
    -------

    .. plot:: ../../examples/polynomial_power_variants/power_function_linear_unit_example.py
       :include-source:

    """
    return torch.where(x >= 0, x, -alpha * (1 - torch.pow(1 + x / beta, beta)))


def power_linear_unit(x: Tensor, alpha: float = 1.0, beta: float = 1.0) -> Tensor:
    r"""Apply the Power Linear Unit activation function.

    .. math::

        \text{PLU}(z) = \begin{cases}
            z, & z \geq 0, \\
            \alpha z^{\beta}, & z < 0,
        \end{cases}

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    alpha : float, optional
        Scale parameter (default 1.0).
    beta : float, optional
        Power parameter (default 1.0).

    Returns
    -------
    torch.Tensor
        The element-wise PLU of the input.


    Source
    ------
    .. seealso::
        A generalization of PReLU described in "Pruning Neural Networks: is it Time to Nip it in the Bud"
        by Bartoldson et al. (2019).

        `arxiv <https://arxiv.org/abs/1910.08489>`_

    Example
    -------


    .. plot:: ../../examples/polynomial_power_variants/power_linear_unit_example.py
       :include-source:

    """
    return torch.where(x >= 0, x, alpha * torch.pow(torch.abs(x), beta))


def inverse_polynomial_linear_unit(x: Tensor, alpha: float = 0.7, beta: float = 0.01) -> Tensor:
    r"""Apply the Inverse Polynomial Linear Unit activation function.

    .. math::

        \text{InvPoLU}(z) = \begin{cases}
            z, & z \geq 0, \\
            \frac{z}{1 + \alpha |z|^{\beta}}, & z < 0,
        \end{cases}

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    alpha : float, optional
        Scale parameter (default 0.7).
    beta : float, optional
        Power parameter (default 0.01).

    Returns
    -------
    torch.Tensor
        The element-wise InvPoLU of the input.


    Source
    ------
    .. seealso::
        Inspired by inverse polynomial functions and described in "InvPoLU: An Inverse Polynomial Activation
        Function for Deep Learning" by Patel et al. (2022).

        `arxiv <https://arxiv.org/abs/2201.12242>`_

    Example
    -------


    .. plot:: ../../examples/polynomial_power_variants/inverse_polynomial_linear_unit_example.py
       :include-source:

    """
    return torch.where(x >= 0, x, x / (1 + alpha * torch.pow(torch.abs(x), beta)))
