"""Other activation functions and their variants for neural networks.

This module provides several additional activation functions that don't fit into
other categories.
"""

import torch
import torch.nn.functional as functional
from torch import Tensor


def abslu(x: Tensor, a: float = 0.01) -> Tensor:
    r"""Apply the Absolute Linear Unit activation function.

    .. math::

        \text{AbsLU}(z) = \begin{cases}
            z, & z \geq 0, \\
            a|z|, & z < 0,
        \end{cases}

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Negative slope (default 0.01).

    Returns
    -------
    torch.Tensor
        The element-wise AbsLU of the input.

    """
    return torch.where(x >= 0, x, a * torch.abs(x))


def mish(x: Tensor) -> Tensor:
    r"""Apply the Mish activation function.

    .. math::

        \text{Mish}(z) = z \cdot \tanh(\text{softplus}(z)) = z \cdot \tanh(\ln(1 + \exp(z)))

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Mish of the input.

    """
    return x * torch.tanh(functional.softplus(x))


def gelu(x: Tensor) -> Tensor:
    r"""Apply the Gaussian Error Linear Unit activation function.

    .. math::

        \text{GELU}(z) = z \cdot \Phi(z)

    Where \Phi(z) is the cumulative distribution function of the standard normal distribution.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise GELU of the input.

    """
    return functional.gelu(x)


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    r"""Apply the Exponential Linear Unit activation function.

    .. math::

        \text{ELU}(z) = \begin{cases}
            z, & z \geq 0, \\
            \alpha(\exp(z) - 1), & z < 0,
        \end{cases}

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    alpha : float, optional
        Scale for the negative factor (default 1.0).

    Returns
    -------
    torch.Tensor
        The element-wise ELU of the input.

    """
    return functional.elu(x, alpha)


def selu(x: Tensor) -> Tensor:
    r"""Apply the Scaled Exponential Linear Unit activation function.

    .. math::

        \text{SELU}(z) = \lambda \begin{cases}
            z, & z \geq 0, \\
            \alpha(\exp(z) - 1), & z < 0,
        \end{cases}

    Where default values are lambda=1.0507 and alpha=1.67326.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise SELU of the input.

    """
    return functional.selu(x)


def celu(x: Tensor, alpha: float = 1.0) -> Tensor:
    r"""Apply the Continuously Differentiable Exponential Linear Unit activation function.

    .. math::

        \text{CELU}(z) = \begin{cases}
            z, & z \geq 0, \\
            \alpha \cdot (\exp(z/\alpha) - 1), & z < 0,
        \end{cases}

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    alpha : float, optional
        Scale for the negative factor (default 1.0).

    Returns
    -------
    torch.Tensor
        The element-wise CELU of the input.

    """
    return functional.celu(x, alpha)


def silu(x: Tensor) -> Tensor:
    r"""Apply the Sigmoid Linear Unit activation function (also known as Swish).

    .. math::

        \text{SiLU}(z) = z \cdot \sigma(z)

    Where \sigma(z) is the sigmoid function.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise SiLU of the input.

    """
    return functional.silu(x)


def swish(x: Tensor) -> Tensor:
    r"""Apply the Swish activation function (same as SiLU).

    .. math::

        \text{Swish}(z) = z \cdot \sigma(z)

    Where \sigma(z) is the sigmoid function.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Swish of the input.

    """
    return functional.silu(x)


def hard_sigmoid(x: Tensor) -> Tensor:
    r"""Apply the Hard Sigmoid activation function.

    .. math::

        \text{HardSigmoid}(z) = \begin{cases}
            0, & z < -3, \\
            1, & z > 3, \\
            z/6 + 1/2, & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Hard Sigmoid of the input.

    """
    return functional.hardsigmoid(x)


def hard_swish(x: Tensor) -> Tensor:
    r"""Apply the Hard Swish activation function.

    .. math::

        \text{HardSwish}(z) = z \cdot \text{HardSigmoid}(z)

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Hard Swish of the input.

    """
    return functional.hardswish(x)


def prelu(x: Tensor, weight: Tensor) -> Tensor:
    r"""Apply the Parametric ReLU activation function.

    .. math::

        \text{PReLU}(z) = \begin{cases}
            z, & z \geq 0, \\
            az, & z < 0,
        \end{cases}

    Where a is a learnable parameter.

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

    """
    return functional.prelu(x, weight)


def tanhsig(x: Tensor) -> Tensor:
    r"""Apply the TanhSig activation function.

    .. math::

        \text{TanhSig}(z) = (z + \tanh(z))\sigma(z)

    Where \sigma(z) is the sigmoid function.

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


def phish(x: Tensor) -> Tensor:
    r"""Apply the Phish activation function.

    .. math::

        \text{Phish}(z) = z \cdot \tanh(\text{GELU}(z))

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Phish of the input.

    """
    return x * torch.tanh(functional.gelu(x))


def exp_swish(x: Tensor) -> Tensor:
    r"""Apply the Exponential Swish activation function.

    .. math::

        \text{ExponentialSwish}(z) = \exp(-z) \cdot \sigma(z)

    Where \sigma(z) is the sigmoid function.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Exponential Swish of the input.

    """
    return torch.exp(-x) * torch.sigmoid(x)


def tanh_linear_unit(x: Tensor) -> Tensor:
    r"""Apply the Tanh Linear Unit activation function.

    .. math::

        \text{TanhLinearUnit}(z) = \begin{cases}
            z, & z \geq 0, \\
            \tanh\left(\frac{z}{2}\right), & z < 0,
        \end{cases}

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

    """
    return torch.maximum(x, x * torch.exp(-torch.abs(x)))


def sin_sig(x: Tensor) -> Tensor:
    r"""Apply the SinSig activation function.

    .. math::

        \text{SinSig}(z) = z \cdot \sin\left(\frac{\pi}{2} \sigma(z)\right)

    Where \sigma(z) is the sigmoid function.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise SinSig of the input.

    """
    return x * torch.sin((torch.pi / 2) * torch.sigmoid(x))


def gish(x: Tensor) -> Tensor:
    r"""Apply the Gish activation function.

    .. math::

        \text{Gish}(z) = z \cdot \ln(2 - \exp(-\exp(z)))

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Gish of the input.

    """
    return x * torch.log(2 - torch.exp(-torch.exp(x)))


def stanhplus(x: Tensor, a: float = 1.5, b: float = 0.5) -> Tensor:
    r"""Apply the Scaled Hyperbolic Tangent activation function.

    .. math::

        \text{STanh}(z) = a \tanh(bz)

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


def penalized_tanh(x: Tensor, a: float = 0.25) -> Tensor:
    r"""Apply the Penalized Hyperbolic Tangent activation function.

    .. math::

        \text{PenalizedHyperbolicTangent}(z) = \begin{cases}
            \tanh(z), & z \geq 0, \\
            \frac{\tanh(z)}{a}, & z < 0,
        \end{cases}

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

    """
    return x * torch.erf(a * torch.exp(b * x))


def complementary_log_log(x: Tensor) -> Tensor:
    r"""Apply the Complementary LogLog activation function.

    .. math::

        \text{ComplementaryLogLog}(z) = 1 - \exp(-\exp(-z))

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Complementary LogLog of the input.

    """
    return 1 - torch.exp(-torch.exp(-x))


def exp_expish(x: Tensor) -> Tensor:
    r"""Apply the ExpExpish activation function.

    .. math::

        \text{ExpExpish}(z) = z \cdot \exp(-\exp(-z))

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise ExpExpish of the input.

    """
    return x * torch.exp(-torch.exp(-x))


def root2sigmoid(x: Tensor) -> Tensor:
    r"""Apply the Root2sigmoid activation function.

    .. math::

        \text{Root2sigmoid}(z) = \frac{\sqrt{2}z}{\sqrt{2^{-2z}} + \sqrt{2^{2z}}}

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Root2sigmoid of the input.

    """
    sqrt2 = torch.sqrt(torch.tensor(2.0))
    return (sqrt2 * x) / (torch.sqrt(torch.pow(2, -2 * x)) + torch.sqrt(torch.pow(2, 2 * x)))


def rootsig(x: Tensor, a: float = 1.0) -> Tensor:
    r"""Apply the Rootsig activation function (also called Unnamed Sigmoid 3).

    .. math::

        \text{Rootsig}(z) = \frac{az}{\sqrt{1 + a^2z^2}}

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Scale parameter (default 1.0).

    Returns
    -------
    torch.Tensor
        The element-wise Rootsig of the input.

    """
    return (a * x) / torch.sqrt(1 + (a * x) ** 2)


def new_sigmoid(x: Tensor) -> Tensor:
    r"""Apply the New Sigmoid activation function.

    .. math::

        \text{NewSigmoid}(z) = \frac{\exp(z) - \exp(-z)}{2(\exp(2z) + \exp(-2z))}

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise New Sigmoid of the input.

    """
    return (torch.exp(x) - torch.exp(-x)) / (2 * (torch.exp(2 * x) + torch.exp(-2 * x)))


def sigmoid_gumbel(x: Tensor) -> Tensor:
    r"""Apply the Sigmoid Gumbel activation function.

    .. math::

        \text{SigmoidGumbel}(z) = \frac{1}{1 + \exp(-z) \exp(-\exp(-z))}

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Sigmoid Gumbel of the input.

    """
    return 1 / (1 + torch.exp(-x) * torch.exp(-torch.exp(-x)))


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

    """
    half_a = a / 2
    return torch.where(
        x < 0, torch.zeros_like(x), torch.where(x <= half_a, x, torch.where(x <= a, a - x, torch.zeros_like(x)))
    )
