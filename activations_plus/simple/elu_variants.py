"""ELU-based activation functions and their variants for neural networks."""

import torch
import torch.nn.functional as functional
from torch import Tensor


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    r"""Apply the Exponential Linear Unit activation function.

    .. math::

        \text{ELU}(z) = \begin{cases}
            z, & z \geq 0, \\
            \alpha(\exp(z) - 1), & z < 0,
        \end{cases}

    .. plot:: ../../examples/elu_variants/elu_example.py
       :include-source:

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

    .. plot:: ../../examples/elu_variants/selu_example.py
       :include-source:

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

    .. plot:: ../../examples/elu_variants/celu_example.py
       :include-source:

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


def abslu(x: Tensor, a: float = 0.01) -> Tensor:
    r"""Apply the Absolute Linear Unit activation function.

    .. math::

        \text{AbsLU}(z) = \begin{cases}
            z, & z \geq 0, \\
            a|z|, & z < 0,
        \end{cases}

    .. plot:: ../../examples/elu_variants/abslu_example.py
       :include-source:

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
