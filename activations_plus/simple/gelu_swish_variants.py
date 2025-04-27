"""GELU and Swish-based activation functions and their variants for neural networks."""

import torch
import torch.nn.functional as functional
from torch import Tensor


def gelu(x: Tensor) -> Tensor:
    r"""Apply the Gaussian Error Linear Unit activation function.

    .. math::

        \text{GELU}(z) = z \cdot \Phi(z)

    Where \Phi(z) is the cumulative distribution function of the standard normal distribution.

    .. plot:: ../../examples/gelu_swish_variants/gelu_example.py
       :include-source:

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


def silu(x: Tensor) -> Tensor:
    r"""Apply the Sigmoid Linear Unit activation function (also known as Swish).

    .. math::

        \text{SiLU}(z) = z \cdot \sigma(z)

    Where \sigma(z) is the sigmoid function.

    .. plot:: ../../examples/gelu_swish_variants/silu_example.py
       :include-source:

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

    .. plot:: ../../examples/gelu_swish_variants/swish_example.py
       :include-source:

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

    .. plot:: ../../examples/gelu_swish_variants/hard_sigmoid_example.py
       :include-source:

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

    .. plot:: ../../examples/gelu_swish_variants/hard_swish_example.py
       :include-source:

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


def mish(x: Tensor) -> Tensor:
    r"""Apply the Mish activation function.

    .. math::

        \text{Mish}(z) = z \cdot \tanh(\text{softplus}(z)) = z \cdot \tanh(\ln(1 + \exp(z)))

    .. plot:: ../../examples/gelu_swish_variants/mish_example.py
       :include-source:

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


def phish(x: Tensor) -> Tensor:
    r"""Apply the Phish activation function.

    .. math::

        \text{Phish}(z) = z \cdot \tanh(\text{GELU}(z))

    .. plot:: ../../examples/gelu_swish_variants/phish_example.py
       :include-source:

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
