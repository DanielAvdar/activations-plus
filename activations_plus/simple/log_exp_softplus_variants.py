"""Log, exp, and softplus-like activations for PyTorch.

This module provides several simple log, exp, and softplus-like activation functions.
"""

import torch
from torch import Tensor


def loglog(x: Tensor) -> Tensor:
    r"""Apply the LogLog activation.

    .. math::

        \mathrm{LogLog}(z) = \exp(-\exp(-z))

    Based on the Gumbel distribution and studied in "Improved Non-Linear Activation Functions in Neural
    Network Applications" by Sibi et al. (2014).

    See: https://www.jatit.org/volumes/Vol67No3/2Vol67No3.pdf

    .. plot:: ../../examples/log_exp_softplus_variants/loglog_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise LogLog of the input.

    """
    return torch.exp(-torch.exp(-x))


def loglogish(x: Tensor) -> Tensor:
    r"""Apply the LogLogish activation.

    .. math::

        \mathrm{LogLogish}(z) = z (1 - \exp(-\exp(z)))

    A variant of LogLog activation inspired by "A Survey of Activation Functions Used in Neural
    Networks" by Bilal et al. (2022).

    See: https://link.springer.com/article/10.1007/s00521-022-07743-y

    .. plot:: ../../examples/log_exp_softplus_variants/loglogish_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise LogLogish of the input.

    """
    return x * (1 - torch.exp(-torch.exp(x)))


def logish(x: Tensor) -> Tensor:
    r"""Apply the Logish activation.

    .. math::

        \mathrm{Logish}(z) = z \log(1 + \sigma(z))

    A logarithmic variant of Swish activation, derived from concepts in "Swish: a Self-Gated Activation
    Function" by Ramachandran et al. (2017).

    See: https://arxiv.org/abs/1710.05941v1

    .. plot:: ../../examples/log_exp_softplus_variants/logish_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Logish of the input.

    """
    s = torch.sigmoid(x)
    return x * torch.log(1 + s)


def soft_exponential(x: Tensor, a: float = 1.0) -> Tensor:
    r"""Apply the Soft Exponential activation (incomplete, placeholder).

    .. math::

        \text{SoftExponential}(z) =
            \begin{cases}
                z, & a = 0 \\
                \exp(z) - 1, & a > 0 \\
                -\log(1-z), & a < 0
            \end{cases}

    Introduced in "A New Activation Function for Artificial Neural Network" by Ushida et al. (2018).

    See: https://arxiv.org/abs/1602.01321

    .. plot:: ../../examples/log_exp_softplus_variants/soft_exponential_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Exponential parameter (default 1.0).

    Returns
    -------
    torch.Tensor
        The element-wise Soft Exponential of the input.

    """
    if a == 0:
        return x
    elif a > 0:
        return torch.exp(x) - 1
    else:
        return -torch.log(1 - x)


def softplus_linear_unit(x: Tensor, a: float = 1.0, b: float = 1.0, c: float = 0.0) -> Tensor:
    r"""Apply the Softplus Linear Unit activation.

    .. math::

        \mathrm{SLU}(z) =
            \begin{cases}
                az, & z \geq 0 \\
                b \log(\exp(z)+1) - c, & z < 0
            \end{cases}

    A generalization of ReLU and Softplus, as described in "Improving Deep Neural Networks with
    Probabilistic Maxout Units" by Sun et al. (2015).

    See: https://arxiv.org/abs/1510.05516

    .. plot:: ../../examples/log_exp_softplus_variants/softplus_linear_unit_example.py
       :include-source:

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Slope for z >= 0 (default 1.0).
    b : float, optional
        Slope for z < 0 (default 1.0).
    c : float, optional
        Offset for z < 0 (default 0.0).

    Returns
    -------
    torch.Tensor
        The element-wise Softplus Linear Unit of the input.

    """
    return torch.where(x >= 0, a * x, b * torch.log(torch.exp(x) + 1) - c)
