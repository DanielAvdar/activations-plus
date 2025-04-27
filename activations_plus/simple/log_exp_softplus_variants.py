"""Log, exp, and softplus-like activations for PyTorch.

This module provides several simple log, exp, and softplus-like activation functions.
"""

import torch


def loglog(x):
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

    """
    return torch.exp(-torch.exp(-x))


def loglogish(x):
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

    """
    return x * (1 - torch.exp(-torch.exp(x)))


def logish(x):
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

    """
    s = torch.sigmoid(x)
    return x * torch.log(1 + s)


def soft_exponential(x, a=1.0):
    r"""Apply the Soft Exponential activation (incomplete, placeholder).

    .. math::

        \text{SoftExponential}(z) =
            \begin{cases}
                z, & a = 0 \\
                \exp(z) - 1, & a > 0 \\
                -\log(1-z), & a < 0
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

    """
    if a == 0:
        return x
    elif a > 0:
        return torch.exp(x) - 1
    else:
        return -torch.log(1 - x)


def softplus_linear_unit(x, a=1.0, b=1.0, c=0.0):
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

    """
    return torch.where(x >= 0, a * x, b * torch.log(torch.exp(x) + 1) - c)
