"""Polynomial and power-based activations for PyTorch.

This module provides several simple polynomial and power-based activation functions.
"""

import torch


def polynomial_linear_unit(x):
    r"""Apply the Polynomial Linear Unit activation.

    .. math::

        \mathrm{PLU}(z) = \begin{cases} z, & z \geq 0 \\ \frac{1}{1-z} - 1, & z < 0 \end{cases}

    .. plot:: ../../examples/polynomial_power_variants/polynomial_linear_unit_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Polynomial Linear Unit of the input.

    """
    return torch.where(x >= 0, x, 1 / (1 - x) - 1)


def power_function_linear_unit(x):
    r"""Apply the Power Function Linear Unit activation.

    .. math::

        \mathrm{PFLU}(z) = z \cdot \frac{1}{2} \left(1 + \frac{z}{\sqrt{1+z^2}}\right)

    .. plot:: ../../examples/polynomial_power_variants/power_function_linear_unit_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Power Function Linear Unit of the input.

    """
    return x * 0.5 * (1 + x / torch.sqrt(1 + x**2))


def power_linear_unit(x, a=1.0):
    r"""Apply the Power Linear Unit activation.

    .. math::

        \mathrm{PowerLU}(z) = \begin{cases} z, & z \geq 0 \\ (1-z)^{-a} - 1, & z < 0 \end{cases}

    .. plot:: ../../examples/polynomial_power_variants/power_linear_unit_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Power parameter (default 1.0).

    Returns
    -------
    torch.Tensor
        The element-wise Power Linear Unit of the input.

    """
    return torch.where(x >= 0, x, (1 - x) ** (-a) - 1)


def inverse_polynomial_linear_unit(x, a=1.0):
    r"""Apply the Inverse Polynomial Linear Unit activation.

    .. math::

        \mathrm{IPLU}(z) = \begin{cases} z, & z \geq 0 \\ \frac{1}{1+|z|^a}, & z < 0 \end{cases}

    .. plot:: ../../examples/polynomial_power_variants/inverse_polynomial_linear_unit_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Power parameter (default 1.0).

    Returns
    -------
    torch.Tensor
        The element-wise Inverse Polynomial Linear Unit of the input.

    """
    return torch.where(x >= 0, x, 1 / (1 + torch.abs(x) ** a))
