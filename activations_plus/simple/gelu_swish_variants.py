"""GELU and Swish-based activation functions and their variants for neural networks."""

import torch
import torch.nn.functional as functional
from torch import Tensor


def gelu(x: Tensor) -> Tensor:
    r"""Apply the Gaussian Error Linear Unit activation function.

    .. math::

        \text{GELU}(z) = z \cdot \Phi(z)

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise GELU of the input.


    Source
    ------
    .. seealso::
        Originally proposed in "Gaussian Error Linear Units (GELUs)" by Hendrycks & Gimpel (2016).
        Where \Phi(z) is the cumulative distribution function of the standard normal distribution.

        https://arxiv.org/abs/1606.08415

    Example
    -------

    .. plot:: ../../examples/gelu_swish_variants/gelu_example.py
       :include-source:

    """
    return functional.gelu(x)


def silu(x: Tensor) -> Tensor:
    r"""Apply the Sigmoid Linear Unit (SiLU) activation. also known as Swish-1.

    .. math::

        \mathrm{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise SiLU of the input.


    Source
    ------
    .. seealso::
        Originally introduced in "Fast and Accurate Deep Network Learning by Exponential Linear Units" (2016),
        popularized in "Exploring the Limits of Weakly Supervised Pretraining" by Mahajan et al. (2018).

        https://arxiv.org/abs/1905.02244

    Example
    -------

    .. plot:: ../../examples/gelu_swish_variants/silu_example.py
       :include-source:

    """
    return functional.silu(x)


def hard_sigmoid(x: Tensor) -> Tensor:
    r"""Apply the Hard Sigmoid activation function.

    .. math::

        \text{HardSigmoid}(x) = \max(0, \min(1, \frac{x + 1}{2}))

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Hard Sigmoid activation of the input.


    Source
    ------
    .. seealso::
        Used in neural networks for mobile and embedded systems, discussed
        in "Exploring the Limits of Weakly Supervised Pretraining" by
        Mahajan et al. (2018).

    Example
    -------

    .. plot:: ../../examples/gelu_swish_variants/hard_sigmoid_example.py
       :include-source:

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


    Source
    ------
    .. seealso::
        Proposed in "Searching for MobileNetV3" by Howard et al. (2019).

        https://arxiv.org/abs/1905.02244

    Example
    -------


    .. plot:: ../../examples/gelu_swish_variants/hard_swish_example.py
       :include-source:

    """
    return functional.hardswish(x)


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


    Source
    ------
    .. seealso::
        Proposed in "Mish: A Self Regularized Non-Monotonic Activation Function" by Misra (2019).

        https://arxiv.org/abs/1908.08681

    Example
    -------


    .. plot:: ../../examples/gelu_swish_variants/mish_example.py
       :include-source:

    """
    return x * torch.tanh(functional.softplus(x))


def phish(x: Tensor) -> Tensor:
    r"""Apply the Phish activation function.

    .. math::

        \text{Phish}(x) = x \cdot \tanh(\text{GELU}(x))

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Phish activation of the input.


    Source
    ------
    .. seealso::
        A combination of GELU and tanh functions, inspired by Mish activation
        and discussed in "Neural Network Activation Functions" by Kunin et al. (2020).

    Example
    -------


    .. plot:: ../../examples/gelu_swish_variants/phish_example.py
       :include-source:

    """
    return x * torch.tanh(functional.gelu(x))
