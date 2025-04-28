"""Sigmoid, tanh, and soft variants for PyTorch.

This module provides several simple sigmoid/tanh-based activation functions.
"""

import torch
from torch import Tensor


def sigmoid(x: Tensor) -> Tensor:
    r"""Apply the standard sigmoid function.

    .. math::

        \sigma(z) = \frac{1}{1 + e^{-z}}

    .. seealso::
        A foundational activation function in neural networks, discussed in depth in "Efficient BackProp"
        by LeCun et al. (1998).

        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise sigmoid of the input.

    Example
    -------
    >>> import torch
    >>> from activations_plus.simple import sigmoid
    >>> x = torch.tensor([-1.0, 0.0, 1.0])
    >>> sigmoid(x)
    tensor([0.2689, 0.5000, 0.7311])

    .. plot:: ../../examples/sigmoid_tanh_variants/sigmoid_example.py
       :include-source:

    """
    return torch.sigmoid(x)


def tanh(x: Tensor) -> Tensor:
    r"""Apply the hyperbolic tangent function.

    .. math::

        \tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}

    .. seealso::
        A classic activation function discussed in "Efficient BackProp" by LeCun et al. (1998).

        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise tanh of the input.

    Example
    -------
    >>> import torch
    >>> from activations_plus.simple import tanh
    >>> x = torch.tensor([-1.0, 0.0, 1.0])
    >>> tanh(x)
    tensor([-0.7616,  0.0000,  0.7616])

    .. plot:: ../../examples/sigmoid_tanh_variants/tanh_example.py
       :include-source:

    """
    return torch.tanh(x)


def hardtanh(x: Tensor, a: float = -1.0, b: float = 1.0) -> Tensor:
    r"""Apply the HardTanh activation (clamps between a and b).

    .. math::

        \mathrm{HardTanh}(z) = \begin{cases} a, & z < a \\ z, & a \leq z \leq b \\ b, & z > b \end{cases}

    .. seealso::
        Described in "Understanding the difficulty of training deep feedforward neural networks"
        by Glorot & Bengio (2010).

        http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

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

    Example
    -------
    >>> import torch
    >>> from activations_plus.simple import hardtanh
    >>> x = torch.tensor([-2.0, -0.5, 0.5, 2.0])
    >>> hardtanh(x)
    tensor([-1.0000, -0.5000,  0.5000,  1.0000])

    .. plot:: ../../examples/sigmoid_tanh_variants/hardtanh_example.py
       :include-source:

    """
    return torch.clamp(x, min=a, max=b)


def softsign(x: Tensor) -> Tensor:
    r"""Apply the Softsign activation.

    .. math::

        \mathrm{Softsign}(z) = \frac{z}{1 + |z|}

    .. seealso::
        Introduced in "Deep Learning via Hessian-free Optimization" by Martens (2010).

        https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Softsign of the input.

    Example
    -------
    >>> import torch
    >>> from activations_plus.simple import softsign
    >>> x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> softsign(x)
    tensor([-0.6667, -0.5000,  0.0000,  0.5000,  0.6667])

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

    .. seealso::
        Proposed in "SQNL: A New Computationally Efficient Activation Function" by Wuraola and Patel (2018).

        https://ieeexplore.ieee.org/document/8489043

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise SQNL of the input.

    Example
    -------
    >>> import torch
    >>> from activations_plus.simple import sqnl
    >>> x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    >>> sqnl(x)
    tensor([-1.0000, -0.7500,  0.0000,  0.7500,  1.0000])

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

    .. seealso::
        First described in "Incorporating Second-Order Functional Knowledge for Better Option Pricing"
        by Dugas et al. (2001).

        https://www.iro.umontreal.ca/~lisa/publications2/index.php/attachments/single/87

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Softplus of the input.

    Example
    -------
    >>> import torch
    >>> from activations_plus.simple import softplus
    >>> x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> softplus(x)
    tensor([0.1269, 0.3133, 0.6931, 1.3133, 2.1269])

    .. plot:: ../../examples/sigmoid_tanh_variants/softplus_example.py
       :include-source:

    """
    return torch.nn.functional.softplus(x)


def tanh_exp(x: Tensor) -> Tensor:
    r"""Apply the TanhExp activation.

    .. math::

        \mathrm{TanhExp}(z) = z \tanh(e^{z})

    .. seealso::
        Introduced in "TanhExp: A Smooth Activation Function with High Convergence Speed for Lightweight
        Neural Networks" by Liu et al. (2020).

        https://arxiv.org/abs/2003.09855

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise TanhExp of the input.

    Example
    -------
    >>> import torch
    >>> from activations_plus.simple import tanh_exp
    >>> x = torch.tensor([-1.0, 0.0, 1.0])
    >>> tanh_exp(x)
    tensor([-0.4621,  0.0000,  0.9640])

    .. plot:: ../../examples/sigmoid_tanh_variants/tanh_exp_example.py
       :include-source:

    """
    return x * torch.tanh(torch.exp(x))


def aria2(x: Tensor) -> Tensor:
    r"""Apply the ARiA2 activation function.

    .. math::

        \mathrm{ARiA2}(z) = \frac{1 + a \tanh^2(z)}{1 + b \tanh^2(z)}

    .. seealso::
        An adaptive rational activation function proposed in "ARiA: Adaptive Rational
        Activation Functions for Convergence Speed for Lightweight Neural Networks"
        by Liu et al. (2020).

        https://arxiv.org/abs/2004.03485

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise ARiA2 of the input.

    Example
    -------

    >>> import torch
    >>> from activations_plus.simple import aria2
    >>> x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> aria2(x)
    tensor([1.3103, 1.2000, 1.0000, 1.2000, 1.3103])

    .. plot:: ../../examples/sigmoid_tanh_variants/aria2_example.py
       :include-source:

    """
    return (1 + 1.5 * torch.tanh(x) ** 2) / (1 + 0.5 * torch.tanh(x) ** 2)
