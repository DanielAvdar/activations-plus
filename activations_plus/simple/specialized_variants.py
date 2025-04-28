"""Specialized activation functions that don't fit into other categories."""

import torch
import torch.nn.functional as functional
from torch import Tensor


def resp(x: Tensor, a: float = 1.0) -> Tensor:
    r"""Apply the Rectified Softplus activation function.

    .. math::

        \text{ReSP}(z) = \begin{cases}
            az + \ln(2), & z \geq 0, \\
            \ln(1 + \exp(z)), & z < 0,
        \end{cases}



    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Slope for positive inputs (default 1.0).

    Returns
    -------
    torch.Tensor
        The element-wise ReSP of the input.


    Source
    ------
    .. seealso::
        A combination of ReLU and softplus discussed in **"Exploring the Relationship: Transformative Adaptive
        Activation Functions in Comparison to Other Activation Functions"** (2024).

        `arxiv <https://arxiv.org/abs/2402.09249>`_

    Example
    -------
    .. plot:: ../../examples/specialized_variants/resp_example.py
       :include-source:

    """
    return torch.where(x >= 0, a * x + torch.log(torch.tensor(2.0)), functional.softplus(x))


def erf_act(x: Tensor, a: float = 0.5, b: float = 1.0) -> Tensor:
    r"""Apply the ErfAct activation function.

    .. math::

        \text{ErfAct}(x) = x \cdot \text{erf}(a \cdot \exp(b \cdot x))


    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Scale parameter (default 0.5).
    b : float, optional
        Exponent parameter (default 1.0).

    Returns
    -------
    torch.Tensor
        The element-wise ErfAct of the input.


    Source
    ------
    .. seealso::
        A variant of activation function using the error function, explored in
        **"ErfAct and Pserf: Non-monotonic Smooth Trainable Activation Functions"** (2022).
        `arxiv <https://arxiv.org/abs/2109.04386>`_

    Example
    -------
    .. plot:: ../../examples/specialized_variants/erf_act_example.py
       :include-source:

    """
    return x * torch.erf(a * torch.exp(b * x))


def hat(x: Tensor, a: float = 1.0) -> Tensor:
    r"""Apply the Hat activation function.

    .. math::

        \text{Hat}(x) = \begin{cases}
            0, & x < 0, \\
            x, & 0 \leq x \leq \frac{a}{2}, \\
            a - x, & \frac{a}{2} \leq x \leq a, \\
            0, & x > a,
        \end{cases}


    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Hat width parameter (default 1.0).

    Returns
    -------
    torch.Tensor
        The element-wise Hat function of the input.


    Source
    ------
    .. seealso::
        Also known as triangular activation function, discussed in
        "On the Activation Function Dependence of the Spectral Bias of Neural Networks"** (2022).

        `arxiv <https://arxiv.org/abs/2208.04924>`_

    Example
    -------
    .. plot:: ../../examples/specialized_variants/hat_example.py
       :include-source:

    """
    half_a = a / 2
    return torch.where(
        x < 0, torch.zeros_like(x), torch.where(x <= half_a, x, torch.where(x <= a, a - x, torch.zeros_like(x)))
    )
