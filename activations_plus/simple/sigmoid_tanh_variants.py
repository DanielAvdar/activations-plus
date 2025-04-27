"""Sigmoid, tanh, and soft variants for PyTorch.

This module provides several simple sigmoid/tanh-based activation functions.
"""

import torch


def sigmoid(x):
    r"""Apply the standard sigmoid function.

    .. math::

        \sigma(z) = \frac{1}{1 + e^{-z}}

    .. plot:: ../../examples/sigmoid_tanh_variants/sigmoid_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise sigmoid of the input.

    """
    return torch.sigmoid(x)


def tanh(x):
    r"""Apply the hyperbolic tangent function.

    .. math::

        \tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}

    .. plot:: ../../examples/sigmoid_tanh_variants/tanh_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise tanh of the input.

    """
    return torch.tanh(x)


def hardtanh(x, a=-1.0, b=1.0):
    r"""Apply the HardTanh activation (clamps between a and b).

    .. math::

        \mathrm{HardTanh}(z) = \begin{cases} a, & z < a \\ z, & a \leq z \leq b \\ b, & z > b \end{cases}

    .. plot:: ../../examples/sigmoid_tanh_variants/hardtanh_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Lower bound (default -1.0).
    b : float, optional
        Upper bound (default 1.0).

    Returns
    -------
    torch.Tensor
        The element-wise HardTanh of the input.

    """
    return torch.clamp(x, min=a, max=b)


def softsign(x):
    r"""Apply the Softsign activation.

    .. math::

        \mathrm{Softsign}(z) = \frac{z}{1 + |z|}

    .. plot:: ../../examples/sigmoid_tanh_variants/softsign_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Softsign of the input.

    """
    return x / (1 + torch.abs(x))


def sqnl(x):
    r"""Apply the SQNL (Square Non-Linear) activation.

    .. math::

        \mathrm{SQNL}(z) = \begin{cases} 1, & z > 2 \\
        z - \frac{z^2}{4}, & 0 \leq z \leq 2 \\
        z + \frac{z^2}{4}, & -2 \leq z < 0 \\
        -1, & z < -2 \end{cases}

    .. plot:: ../../examples/sigmoid_tanh_variants/sqnl_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise SQNL of the input.

    """
    return torch.where(
        x > 2,
        torch.ones_like(x),
        torch.where(
            (x >= 0) & (x <= 2), x - (x**2) / 4, torch.where((x >= -2) & (x < 0), x + (x**2) / 4, -torch.ones_like(x))
        ),
    )


def softplus(x):
    r"""Apply the Softplus activation.

    .. math::

        \mathrm{Softplus}(z) = \log(1 + e^{z})

    .. plot:: ../../examples/sigmoid_tanh_variants/softplus_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Softplus of the input.

    """
    return torch.nn.functional.softplus(x)


def tanh_exp(x):
    r"""Apply the TanhExp activation.

    .. math::

        \mathrm{TanhExp}(z) = z \tanh(e^{z})

    .. plot:: ../../examples/sigmoid_tanh_variants/tanh_exp_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise TanhExp of the input.

    """
    return x * torch.tanh(torch.exp(x))
