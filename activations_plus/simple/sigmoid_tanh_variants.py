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

        `arxiv <https://arxiv.org/abs/cs/9806004>`_

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

        `arxiv <https://arxiv.org/abs/1602.02830>`_

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

        `arxiv <https://arxiv.org/abs/1003.0358>`_

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

        `arxiv <https://arxiv.org/abs/1006.0443>`_

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

        `arxiv <https://arxiv.org/abs/1803.07318>`_

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

        `arxiv <https://arxiv.org/abs/1206.4104>`_

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

        `arxiv <https://arxiv.org/abs/2003.09855>`_

    Example
    -------


    .. plot:: ../../examples/sigmoid_tanh_variants/tanh_exp_example.py
       :include-source:

    """
    return x * torch.tanh(torch.exp(x))


def aria2(x: Tensor, alpha: float = 1.5, beta: float = 0.5) -> Tensor:
    r"""Apply the ARiA2 activation function based on Richard's curve.

    .. math::

        \mathrm{ARiA2}(z) = \frac{1}{(1 + e^{-\alpha z})^{1/\beta}}


    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    alpha : float, optional
        Alpha parameter controlling the steepness (default 1.5).
    beta : float, optional
        Beta parameter controlling the asymptotic behavior (default 0.5).

    Returns
    -------
    torch.Tensor
        The element-wise ARiA2 of the input.


    Source
    ------
    .. seealso::
        Introduced in "ARiA: Utilizing Richard's Curve for Controlling the Non-monotonicity
        of the Activation Function in Deep Neural Nets".

        `arxiv <https://arxiv.org/abs/1805.08878>`_

    Example
    -------

    .. plot:: ../../examples/sigmoid_tanh_variants/aria2_example.py
       :include-source:

    """
    return torch.pow(1 + torch.exp(-alpha * x), -1 / beta)


def isru(x: Tensor, alpha: float = 1.0) -> Tensor:
    r"""Apply the Inverse Square Root Unit activation function.

    .. math::

        \text{ISRU}(z) = \frac{z}{\sqrt{1 + \alpha z^2}}

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    alpha : float, optional
        Scaling parameter (default 1.0).

    Returns
    -------
    torch.Tensor
        The element-wise ISRU of the input.


    Source
    ------
    .. seealso::
        Proposed in "Improving Deep Neural Networks with New Activation Functions"
        by Carlile et al. (2017).

        `arxiv <https://arxiv.org/abs/1710.09967>`_

    Example
    -------

    .. plot:: ../../examples/sigmoid_tanh_variants/isru_example.py
       :include-source:

    """
    return x / torch.sqrt(1 + alpha * x**2)
