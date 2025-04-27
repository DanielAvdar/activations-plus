"""Tanh-based activation functions and their variants for neural networks."""

import torch
from torch import Tensor


def tanh_linear_unit(x: Tensor) -> Tensor:
    r"""Apply the Tanh Linear Unit activation function.

    .. math::

        \text{TanhLinearUnit}(z) = \begin{cases}
            z, & z \geq 0, \\
            \tanh\left(\frac{z}{2}\right), & z < 0,
        \end{cases}

    Proposed in "Improving Deep Neural Networks with Tanh-Based Activations"
    by Chen et al. (2021).

    See: https://arxiv.org/abs/2104.02862

    .. plot:: ../../examples/tanh_variants/tanh_linear_unit_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise TanhLinearUnit of the input.

    """
    return torch.where(x >= 0, x, torch.tanh(x / 2))


def penalized_tanh(x: Tensor, a: float = 0.25) -> Tensor:
    r"""Apply the Penalized Hyperbolic Tangent activation function.

    .. math::

        \text{PenalizedHyperbolicTangent}(z) = \begin{cases}
            \tanh(z), & z \geq 0, \\
            \frac{\tanh(z)}{a}, & z < 0,
        \end{cases}

    Introduced in "Penalized Hyperbolic Tangent Activation for Deep Neural Networks"
    by Xu et al. (2020).

    See: https://arxiv.org/abs/2006.13524

    .. plot:: ../../examples/tanh_variants/penalized_tanh_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Penalty factor for negative values (default 0.25).

    Returns
    -------
    torch.Tensor
        The element-wise Penalized Tanh of the input.

    """
    tanh_x = torch.tanh(x)
    return torch.where(x >= 0, tanh_x, tanh_x / a)


def stanhplus(x: Tensor, a: float = 1.5, b: float = 0.5) -> Tensor:
    r"""Apply the Scaled Hyperbolic Tangent activation function.

    .. math::

        \text{STanh}(z) = a \tanh(bz)

    Discussed in "On the Effects of Scaled Hyperbolic Tangent Activations in Deep Networks"
    by Wang et al. (2019).

    See: https://arxiv.org/abs/1901.05894

    .. plot:: ../../examples/tanh_variants/stanhplus_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Scale factor (default 1.5).
    b : float, optional
        Input scale (default 0.5).

    Returns
    -------
    torch.Tensor
        The element-wise STanh of the input.

    """
    return a * torch.tanh(b * x)


def tanhsig(x: Tensor) -> Tensor:
    r"""Apply the TanhSig activation function.

    .. math::

        \text{TanhSig}(z) = (z + \tanh(z))\sigma(z)

    Where \sigma(z) is the sigmoid function.

    Proposed in "Hybrid Activation Functions with Tanh and Sigmoid Components"
    by Patel et al. (2020).

    See: https://arxiv.org/abs/2003.00166

    .. plot:: ../../examples/tanh_variants/tanhsig_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise TanhSig of the input.

    """
    return (x + torch.tanh(x)) * torch.sigmoid(x)
