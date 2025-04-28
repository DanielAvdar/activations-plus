"""Log, exp, and softplus-like activations for PyTorch.

This module provides several simple log, exp, and softplus-like activation functions.
"""

import torch
from torch import Tensor


def loglog(x: Tensor) -> Tensor:
    r"""Apply the LogLog activation.

    .. math::

        \mathrm{LogLog}(z) = \exp(-\exp(-z))

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise LogLog of the input.


    Source
    ------
    .. seealso::
        Based on the Gumbel distribution and studied in "Improved Non-Linear Activation Functions in Neural
        Network Applications" by Sibi et al. (2014).

        https://www.jatit.org/volumes/Vol67No3/2Vol67No3.pdf

    Example
    -------
    .. plot:: ../../examples/log_exp_softplus_variants/loglog_example.py
       :include-source:

    """
    return torch.exp(-torch.exp(-x))


def loglogish(x: Tensor) -> Tensor:
    r"""Apply the LogLogish activation.

    .. math::

        \mathrm{LogLogish}(z) = z (1 - \exp(-\exp(z)))

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise LogLogish of the input.


    Source
    ------
    .. seealso::
        A variant of LogLog activation inspired by "A Survey of Activation Functions Used in Neural
        Networks" by Bilal et al. (2022).

        https://link.springer.com/article/10.1007/s00521-022-07743-y

    Example
    -------

    .. plot:: ../../examples/log_exp_softplus_variants/loglogish_example.py
       :include-source:

    """
    return x * (1 - torch.exp(-torch.exp(x)))


def logish(x: Tensor) -> Tensor:
    r"""Apply the Logish activation.

    .. math::

        \mathrm{Logish}(z) = z \log(1 + \sigma(z))

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Logish of the input.


    Source
    ------
    .. seealso::
        A logarithmic variant of Swish activation, derived from concepts in "Swish: a Self-Gated Activation
        Function" by Ramachandran et al. (2017).

        https://arxiv.org/abs/1710.05941v1

    Example
    -------


    .. plot:: ../../examples/log_exp_softplus_variants/logish_example.py
       :include-source:

    """
    s = torch.sigmoid(x)
    return x * torch.log(1 + s)


def soft_exponential(x: Tensor, a: float = 1.0) -> Tensor:
    r"""Apply the Soft Exponential activation function.

    .. math::

        \text{SoftExponential}(z) =
            \begin{cases}
                -\frac{\log(1 - a(z + a))}{a}, & a < 0 \\
                z, & a = 0 \\
                \frac{e^{az} - 1}{a} + a, & a > 0
            \end{cases}

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


    Source
    ------
    .. seealso::
        Introduced in "A continuum among logarithmic, linear, and exponential functions, and
        its potential to improve generalization in neural networks" by Godfrey and Gashler (2016).

        https://arxiv.org/abs/1602.01321

    Example
    -------

    .. plot:: ../../examples/log_exp_softplus_variants/soft_exponential_example.py
       :include-source:

    """
    if a == 0:
        return x
    elif a > 0:
        return (torch.exp(a * x) - 1) / a + a
    else:  # a < 0
        return -torch.log(1 - a * (x + a)) / a


def softplus_linear_unit(x: Tensor, a: float = 1.0, b: float = 1.0, c: float = 0.0) -> Tensor:
    r"""Apply the Softplus Linear Unit activation.

    .. math::

        \mathrm{SLU}(z) =
            \begin{cases}
                az, & z \geq 0 \\
                b \log(\exp(z)+1) - c, & z < 0
            \end{cases}

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


    Source
    ------
    .. seealso::
        A generalization of ReLU and Softplus, as described in "Improving Deep Neural Networks with
        Probabilistic Maxout Units" by Sun et al. (2015).

        https://arxiv.org/abs/1510.05516

    Example
    -------

    .. plot:: ../../examples/log_exp_softplus_variants/softplus_linear_unit_example.py
       :include-source:

    """
    return torch.where(x >= 0, a * x, b * torch.log(torch.exp(x) + 1) - c)
