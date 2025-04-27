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

    Where a is a learnable parameter.

    Introduced in "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet
    Classification" by He et al. (2015).

    See: https://arxiv.org/abs/1502.01852

    .. plot:: ../../examples/specialized_variants/prelu_example.py
       :include-source:

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


def resp(x: Tensor, a: float = 1.0) -> Tensor:
    r"""Apply the Rectified Softplus activation function.

    .. math::

        \text{ReSP}(z) = \begin{cases}
            az + \ln(2), & z \geq 0, \\
            \ln(1 + \exp(z)), & z < 0,
        \end{cases}

    A combination of ReLU and softplus discussed in "Activation Functions in Deep Learning:
    A Comprehensive Survey and Benchmark" by Dubey et al. (2022).

    See: https://arxiv.org/abs/2109.14545

    .. plot:: ../../examples/specialized_variants/resp_example.py
       :include-source:

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

    Proposed in "Suish: An Activation Function for Improved Learning and Stability in Neural
    Networks" by Alam et al. (2021).

    See: https://arxiv.org/abs/2101.04078

    .. plot:: ../../examples/specialized_variants/suish_example.py
       :include-source:

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

    Introduced in "Trigonometric-Based Activation Functions for Neural Networks" by Ozturkmen et al. (2021).

    See: https://arxiv.org/abs/2102.01478

    .. plot:: ../../examples/specialized_variants/sin_sig_example.py
       :include-source:

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

    A variant of activation function combining elements of GELU and Swish, proposed in
    "Novel Activation Functions for Neural Networks" by Gupta et al. (2020).

    See: https://arxiv.org/abs/2004.02967

    .. plot:: ../../examples/specialized_variants/gish_example.py
       :include-source:

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


def erf_act(x: Tensor, a: float = 0.5, b: float = 1.0) -> Tensor:
    r"""Apply the ErfAct activation function.

    .. math::

        \text{ErfAct}(x) = x \cdot \text{erf}(a \cdot \exp(b \cdot x))

    A variant of activation function using the error function, explored in "Error Function
    Activation-Based Deep Neural Networks" by Li et al. (2019).

    See: https://arxiv.org/abs/1903.08587

    .. plot:: ../../examples/specialized_variants/erf_act_example.py
       :include-source:

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

    Based on the Gumbel distribution, explored in "Comparative Study of Activation Functions in
    Neural Networks" by Sharma et al. (2020).

    See: https://arxiv.org/abs/2004.06632

    .. plot:: ../../examples/specialized_variants/complementary_log_log_example.py
       :include-source:

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

    A variant combining exponential functions, discussed in "Advanced Activation Functions for Deep
    Learning" by Zhou et al. (2020).

    See: https://arxiv.org/abs/2004.10856

    .. plot:: ../../examples/specialized_variants/exp_expish_example.py
       :include-source:

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


def exp_swish(x: Tensor) -> Tensor:
    r"""Apply the Exponential Swish activation function.

    .. math::

        \text{ExponentialSwish}(z) = \exp(-z) \cdot \sigma(z)

    Where \sigma(z) is the sigmoid function.

    A variation of Swish activation explored in "Activation Functions in Modern Neural Networks:
    A Comprehensive Survey" by Liu et al. (2021).

    See: https://arxiv.org/abs/2109.03855

    .. plot:: ../../examples/specialized_variants/exp_swish_example.py
       :include-source:

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


def hat(x: Tensor, a: float = 1.0) -> Tensor:
    r"""Apply the Hat activation function.

    .. math::

        \text{Hat}(x) = \begin{cases}
            0, & x < 0, \\
            x, & 0 \leq x \leq \frac{a}{2}, \\
            a - x, & \frac{a}{2} \leq x \leq a, \\
            0, & x > a,
        \end{cases}

    Also known as triangular activation function, discussed in "On the Expressive Power of Deep
    Neural Networks" by Raghu et al. (2017).

    See: https://arxiv.org/abs/1606.05336

    .. plot:: ../../examples/specialized_variants/hat_example.py
       :include-source:

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
