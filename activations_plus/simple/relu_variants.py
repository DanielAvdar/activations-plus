"""ReLU and Leaky ReLU variants for PyTorch.

This module provides several simple ReLU-based activation functions.
"""

import torch


def relu(x):
    r"""Apply the Rectified Linear Unit activation.

    .. math::

        \mathrm{ReLU}(z) = \max(0, z)

    .. plot::
       :include-source:

       from activations_plus.simple.relu_variants import relu
       import torch
       import matplotlib.pyplot as plt
       x = torch.linspace(-3, 3, 200)
       y = relu(x)
       plt.plot(x.numpy(), y.numpy())
       plt.title("ReLU")
       plt.xlabel("Input")
       plt.ylabel("Output")
       plt.grid(alpha=0.3)
       plt.show()

    Returns
    -------
    torch.Tensor
        The element-wise ReLU of the input.

    """
    return torch.relu(x)


def lrelu(x, a=0.01):
    r"""Apply the Leaky ReLU activation.

    .. math::

        \mathrm{LReLU}(z) = \begin{cases} z, & z \geq 0 \\ \frac{z}{a}, & z < 0 \end{cases}

    .. plot::
       :include-source:

       from activations_plus.simple.relu_variants import lrelu
       import torch
       import matplotlib.pyplot as plt
       x = torch.linspace(-3, 3, 200)
       y = lrelu(x, a=0.1)
       plt.plot(x.numpy(), y.numpy())
       plt.title("Leaky ReLU (a=0.1)")
       plt.xlabel("Input")
       plt.ylabel("Output")
       plt.grid(alpha=0.3)
       plt.show()

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Negative slope (default 0.01).

    Returns
    -------
    torch.Tensor
        The element-wise Leaky ReLU of the input.

    """
    return torch.where(x >= 0, x, x / a)


def blrelu(x, a=0.01, b=1.0, c=0.0):
    r"""Apply the Bounded Leaky ReLU activation.

    .. math::

        \mathrm{BLReLU}(z) = \begin{cases} az, & z \leq 0 \\ z, & 0 < z < b \\ az + c, & z \geq b \end{cases}

    .. plot::
       :include-source:

       from activations_plus.simple.relu_variants import blrelu
       import torch
       import matplotlib.pyplot as plt
       x = torch.linspace(-3, 3, 200)
       y = blrelu(x)
       plt.plot(x.numpy(), y.numpy())
       plt.title("Bounded Leaky ReLU")
       plt.xlabel("Input")
       plt.ylabel("Output")
       plt.grid(alpha=0.3)
       plt.show()

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Negative slope (default 0.01).
    b : float, optional
        Upper bound (default 1.0).
    c : float, optional
        Offset for z >= b (default 0.0).

    Returns
    -------
    torch.Tensor
        The element-wise Bounded Leaky ReLU of the input.

    """
    return torch.where(x <= 0, a * x, torch.where((x > 0) & (x < b), x, a * x + c))


def rrelu(x, a=0.01):
    r"""Apply the Randomized Leaky ReLU activation.

    .. math::

        \mathrm{RReLU}(z) = \begin{cases} z, & z \geq 0 \\ z a, & z < 0 \end{cases}

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Negative slope (default 0.01).

    Returns
    -------
    torch.Tensor
        The element-wise Randomized Leaky ReLU of the input.

    """
    return torch.where(x >= 0, x, x * a)


def trec(x, a=0.0):
    r"""Apply the Truncated Rectified activation.

    .. math::

        \mathrm{TRec}(z) = \begin{cases} z, & z > a \\ 0, & z \leq a \end{cases}

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Truncation threshold (default 0.0).

    Returns
    -------
    torch.Tensor
        The element-wise Truncated Rectified activation of the input.

    """
    return torch.where(x > a, x, torch.zeros_like(x))


def dual_line(x, a=1.0, b=0.01, m=0.0):
    r"""Apply the Dual Line activation.

    .. math::

        \mathrm{DualLine}(x) = \begin{cases} a x + m, & x \geq 0 \\ b x + m, & x < 0 \end{cases}

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Slope for x >= 0 (default 1.0).
    b : float, optional
        Slope for x < 0 (default 0.01).
    m : float, optional
        Offset (default 0.0).

    Returns
    -------
    torch.Tensor
        The element-wise Dual Line activation of the input.

    """
    return torch.where(x >= 0, a * x + m, b * x + m)


def mrelu(x):
    r"""Apply the Mirrored ReLU activation.

    .. math::

        \mathrm{mReLU}(z) = \min(\mathrm{ReLU}(1-z), \mathrm{ReLU}(1+z))

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Mirrored ReLU of the input.

    """
    return torch.min(torch.relu(1 - x), torch.relu(1 + x))
