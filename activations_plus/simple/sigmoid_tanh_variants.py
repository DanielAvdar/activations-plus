"""Sigmoid, tanh, and soft variants for PyTorch.

This module provides several simple sigmoid/tanh-based activation functions.
"""

import torch
from torch import Tensor


def sigmoid(x: Tensor) -> Tensor:
    r"""Apply the standard sigmoid function.

    .. math::

        \sigma(z) = \frac{1}{1 + e^{-z}}




    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise sigmoid of the input.


    Source
    ------
    .. seealso::
        A foundational activation function in neural networks, discussed in depth in "Efficient BackProp"
        by LeCun et al. (1998).

        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

    Example
    -------


    .. plot:: ../../examples/sigmoid_tanh_variants/sigmoid_example.py
       :include-source:

    """
    return torch.sigmoid(x)


def tanh(x: Tensor) -> Tensor:
    r"""Apply the hyperbolic tangent function.

    .. math::

        \text{tanh}(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}




    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise hyperbolic tangent of the input.


    Source
    ------
    .. seealso::
        A classic activation function with origins in statistical mechanics, described in "Neural Networks
        for Pattern Recognition" by Bishop (1995).


    Example
    -------


    .. plot:: ../../examples/sigmoid_tanh_variants/tanh_example.py
       :include-source:

    """
    return torch.tanh(x)


def hardtanh(x: Tensor, a: float = -1.0, b: float = 1.0) -> Tensor:
    r"""Apply the HardTanh activation (clamps between a and b).

    .. math::

        \mathrm{HardTanh}(z) = \begin{cases} a, & z < a \\ z, & a \leq z \leq b \\ b, & z > b \end{cases}


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


    Source
    ------
    .. seealso::
        Described in "Understanding the difficulty of training deep feedforward neural networks"
        by Glorot & Bengio (2010).

        http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

    Example
    -------


    .. plot:: ../../examples/sigmoid_tanh_variants/hardtanh_example.py
       :include-source:

    """
    return torch.clamp(x, min=a, max=b)


def softsign(x: Tensor) -> Tensor:
    r"""Apply the Softsign activation.

    .. math::

        \mathrm{Softsign}(z) = \frac{z}{1 + |z|}



    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Softsign of the input.


    Source
    ------
    .. seealso::
        Introduced in "Deep Learning via Hessian-free Optimization" by Martens (2010).

        https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf

    Example
    -------


    .. plot:: ../../examples/sigmoid_tanh_variants/softsign_example.py
       :include-source:

    """
    return x / (1 + torch.abs(x))


def sqnl(x: Tensor) -> Tensor:
    r"""Apply the SQNL (Square Non-Linear) activation.

    .. math::

        \mathrm{SQNL}(z) = \begin{cases} 1, & z > 2 \\
        z - \frac{z^2}{4}, & 0 \leq z \leq 2 \\
        z + \frac{z^2}{4}, & -2 \leq z < 0 \\
        -1, & z < -2 \end{cases}



    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise SQNL of the input.


    Source
    ------
    .. seealso::
        Proposed in "SQNL: A New Computationally Efficient Activation Function" by Wuraola and Patel (2018).

        https://ieeexplore.ieee.org/document/8489043

    Example
    -------

    .. plot:: ../../examples/sigmoid_tanh_variants/sqnl_example.py
       :include-source:

    """
    return torch.where(
        x > 2,
        torch.ones_like(x),
        torch.where(
            (x >= 0) & (x <= 2), x - (x**2) / 4, torch.where((x >= -2) & (x < 0), x + (x**2) / 4, -torch.ones_like(x))
        ),
    )


def softplus(x: Tensor) -> Tensor:
    r"""Apply the Softplus activation.

    .. math::

        \mathrm{Softplus}(z) = \log(1 + e^{z})


    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Softplus of the input.


    Source
    ------
    .. seealso::
        First described in "Incorporating Second-Order Functional Knowledge for Better Option Pricing"
        by Dugas et al. (2001).

        https://www.iro.umontreal.ca/~lisa/publications2/index.php/attachments/single/87

    Example
    -------
    .. plot:: ../../examples/sigmoid_tanh_variants/softplus_example.py
       :include-source:

    """
    return torch.nn.functional.softplus(x)


def tanh_exp(x: Tensor) -> Tensor:
    r"""Apply the TanhExp activation.

    .. math::

        \mathrm{TanhExp}(z) = z \tanh(e^{z})


    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise TanhExp of the input.


    Source
    ------
    .. seealso::
        Introduced in "TanhExp: A Smooth Activation Function with High Convergence Speed for Lightweight
        Neural Networks" by Liu et al. (2020).

        https://arxiv.org/abs/2003.09855

    Example
    -------


    .. plot:: ../../examples/sigmoid_tanh_variants/tanh_exp_example.py
       :include-source:

    """
    return x * torch.tanh(torch.exp(x))


def aria2(x: Tensor) -> Tensor:
    r"""Apply the ARiA2 activation function.

    .. math::

        \mathrm{ARiA2}(z) = \frac{1 + a \tanh^2(z)}{1 + b \tanh^2(z)}


    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise ARiA2 of the input.


    Source
    ------
    .. seealso::
        An adaptive rational activation function proposed in "ARiA: Adaptive Rational
        Activation Functions for Convergence Speed for Lightweight Neural Networks"
        by Liu et al. (2020).

        https://arxiv.org/abs/2004.03485

    Example
    -------



    .. plot:: ../../examples/sigmoid_tanh_variants/aria2_example.py
       :include-source:

    """
    return (1 + 1.5 * torch.tanh(x) ** 2) / (1 + 0.5 * torch.tanh(x) ** 2)
