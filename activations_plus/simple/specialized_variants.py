"""Specialized activation functions that don't fit into other categories."""

import torch
import torch.nn.functional as functional
from torch import Tensor


def prelu(x: Tensor, weight: Tensor) -> Tensor:
    r"""Apply the Parametric ReLU activation function.

    .. math::

        \text{PReLU}(z) = \begin{cases}
            z, & z \geq 0, \\
            az, & z < 0,
        \end{cases}



    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    weight : torch.Tensor
        The learnable parameter for the negative slope.

    Returns
    -------
    torch.Tensor
        The element-wise PReLU of the input.


    Source
    ------
    .. seealso::
        Introduced in "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet
        Classification" by He et al. (2015).

        Where a is a learnable parameter.

        `arxiv <https://arxiv.org/abs/1502.01852>`_

    Example
    -------
    .. plot:: ../../examples/specialized_variants/prelu_example.py
       :include-source:

    """
    return functional.prelu(x, weight)


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


def suish(x: Tensor) -> Tensor:
    r"""Apply the Suish activation function.

    .. math::

        \text{Suish}(z) = \max(z, z \cdot \exp(-|z|))


    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Suish of the input.


    Source
    ------
    .. seealso::
        Proposed in "Suish: An Activation Function for Improved Learning and Stability in Neural
        Networks" by Alam et al. (2021).

        `arxiv <https://arxiv.org/abs/2101.04078>`_

    Example
    -------
    .. plot:: ../../examples/specialized_variants/suish_example.py
       :include-source:

    """
    return torch.maximum(x, x * torch.exp(-torch.abs(x)))


def sin_sig(x: Tensor) -> Tensor:
    r"""Apply the SinSig activation function.

    .. math::

        \text{SinSig}(z) = z \cdot \sin\left(\frac{\pi}{2} \sigma(z)\right)


    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise SinSig of the input.


    Source
    ------
    .. seealso::
        Introduced in "Trigonometric-Based Activation Functions for Neural Networks"
        by Ozturkmen et al. (2021).

        Where \sigma(z) is the sigmoid function.

        `arxiv <https://arxiv.org/abs/2102.01478>`_

    Example
    -------
    .. plot:: ../../examples/specialized_variants/sin_sig_example.py
       :include-source:

    """
    return x * torch.sin((torch.pi / 2) * torch.sigmoid(x))


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
        **"ErfAct and Pserf: Non-monotonic Smooth Trainable Activation Functions"** (2019).
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
        The element-wise Hat of the input.


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
