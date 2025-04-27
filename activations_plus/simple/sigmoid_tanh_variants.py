"""Sigmoid, tanh, and soft variants for PyTorch.

This module provides several simple sigmoid/tanh-based activation functions.
"""

import torch


def sigmoid(x):
    r"""Apply the standard sigmoid function.

    .. math::

        \sigma(z) = \frac{1}{1 + e^{-z}}

    .. plot::
       :include-source:

       from activations_plus.simple.sigmoid_tanh_variants import sigmoid
       import torch
       import matplotlib.pyplot as plt
       x = torch.linspace(-6, 6, 200)
       y = sigmoid(x)
       plt.plot(x.numpy(), y.numpy())
       plt.title("Sigmoid")
       plt.xlabel("Input")
       plt.ylabel("Output")
       plt.grid(alpha=0.3)
       plt.show()

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise sigmoid of the input.

    """
    return torch.sigmoid(x)


def tanh(x):
    r"""Apply the hyperbolic tangent function.

    .. math::

        \tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}

    .. plot::
       :include-source:

       from activations_plus.simple.sigmoid_tanh_variants import tanh
       import torch
       import matplotlib.pyplot as plt
       x = torch.linspace(-6, 6, 200)
       y = tanh(x)
       plt.plot(x.numpy(), y.numpy())
       plt.title("Tanh")
       plt.xlabel("Input")
       plt.ylabel("Output")
       plt.grid(alpha=0.3)
       plt.show()

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise tanh of the input.

    """
    return torch.tanh(x)


def hardtanh(x, a=-1.0, b=1.0):
    r"""Apply the HardTanh activation (clamps between a and b).

    .. math::

        \mathrm{HardTanh}(z) = \begin{cases} a, & z < a \\ z, & a \leq z \leq b \\ b, & z > b \end{cases}

    .. plot::
       :include-source:

       from activations_plus.simple.sigmoid_tanh_variants import hardtanh
       import torch
       import matplotlib.pyplot as plt
       x = torch.linspace(-6, 6, 200)
       y = hardtanh(x)
       plt.plot(x.numpy(), y.numpy())
       plt.title("HardTanh")
       plt.xlabel("Input")
       plt.ylabel("Output")
       plt.grid(alpha=0.3)
       plt.show()

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Lower bound (default -1.0).
    b : float, optional
        Upper bound (default 1.0).

    Returns
    -------
    torch.Tensor
        The element-wise HardTanh of the input.

    """
    return torch.clamp(x, min=a, max=b)


def softsign(x):
    r"""Apply the Softsign activation.

    .. math::

        \mathrm{Softsign}(z) = \frac{z}{1 + |z|}

    .. plot::
       :include-source:

       from activations_plus.simple.sigmoid_tanh_variants import softsign
       import torch
       import matplotlib.pyplot as plt
       x = torch.linspace(-6, 6, 200)
       y = softsign(x)
       plt.plot(x.numpy(), y.numpy())
       plt.title("Softsign")
       plt.xlabel("Input")
       plt.ylabel("Output")
       plt.grid(alpha=0.3)
       plt.show()

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Softsign of the input.

    """
    return x / (1 + torch.abs(x))


def sqnl(x):
    r"""Apply the SQNL (Square Non-Linear) activation.

    .. math::

        \mathrm{SQNL}(z) = \begin{cases} 1, & z > 2 \\
        z - \frac{z^2}{4}, & 0 \leq z \leq 2 \\
        z + \frac{z^2}{4}, & -2 \leq z < 0 \\
        -1, & z < -2 \end{cases}

    .. plot::
       :include-source:

       from activations_plus.simple.sigmoid_tanh_variants import sqnl
       import torch
       import matplotlib.pyplot as plt
       x = torch.linspace(-3, 3, 200)
       y = sqnl(x)
       plt.plot(x.numpy(), y.numpy())
       plt.title("SQNL")
       plt.xlabel("Input")
       plt.ylabel("Output")
       plt.grid(alpha=0.3)
       plt.show()

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise SQNL of the input.

    """
    return torch.where(
        x > 2,
        torch.ones_like(x),
        torch.where(
            (x >= 0) & (x <= 2), x - (x**2) / 4, torch.where((x >= -2) & (x < 0), x + (x**2) / 4, -torch.ones_like(x))
        ),
    )


def softplus(x):
    r"""Apply the Softplus activation.

    .. math::

        \mathrm{Softplus}(z) = \log(1 + e^{z})

    .. plot::
       :include-source:

       from activations_plus.simple.sigmoid_tanh_variants import softplus
       import torch
       import matplotlib.pyplot as plt
       x = torch.linspace(-6, 6, 200)
       y = softplus(x)
       plt.plot(x.numpy(), y.numpy())
       plt.title("Softplus")
       plt.xlabel("Input")
       plt.ylabel("Output")
       plt.grid(alpha=0.3)
       plt.show()

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Softplus of the input.

    """
    return torch.nn.functional.softplus(x)


def tanh_exp(x):
    r"""Apply the TanhExp activation.

    .. math::

        \mathrm{TanhExp}(z) = z \tanh(e^{z})

    .. plot::
       :include-source:

       from activations_plus.simple.sigmoid_tanh_variants import tanh_exp
       import torch
       import matplotlib.pyplot as plt
       x = torch.linspace(-3, 3, 200)
       y = tanh_exp(x)
       plt.plot(x.numpy(), y.numpy())
       plt.title("TanhExp")
       plt.xlabel("Input")
       plt.ylabel("Output")
       plt.grid(alpha=0.3)
       plt.show()

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise TanhExp of the input.

    """
    return x * torch.tanh(torch.exp(x))
