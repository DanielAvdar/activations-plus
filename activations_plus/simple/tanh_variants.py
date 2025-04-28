"""Tanh-based activation functions and their variants for neural networks."""

import torch
from torch import Tensor


def penalized_tanh(x: Tensor, a: float = 0.25) -> Tensor:
    r"""Apply the Penalized Tanh activation function.

    .. math::

        \text{PenalizedTanh}(x) = \tanh(x) - a \cdot \tanh^2(x)

    .. seealso::
        A variant of tanh activation function proposed in "Activation Functions: Comparison in Neural
        Network Architecture" by Sharma et al. (2021).

        https://arxiv.org/abs/2109.14545

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Penalty coefficient (default 0.25).

    Returns
    -------
    torch.Tensor
        The element-wise Penalized Tanh of the input.

    Example
    -------
    >>> import torch
    >>> from activations_plus.simple import penalized_tanh
    >>> x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> penalized_tanh(x, a=0.25)
    tensor([-0.8197, -0.6379,  0.0000,  0.6379,  0.8197])

    .. plot:: ../../examples/tanh_variants/penalized_tanh_example.py
       :include-source:

    """
    t = torch.tanh(x)
    return t - a * t**2


def tanh_linear_unit(x: Tensor, a: float = 0.25) -> Tensor:
    r"""Apply the Tanh Linear Unit activation function.

    .. math::

        \text{TLU}(x) = \begin{cases}
            x, & x \geq 0, \\
            \tanh(x), & x < 0,
        \end{cases}

    .. seealso::
        Combines tanh and linear functions, proposed in "TanhSoft: A Smooth Activation Function
        with High Convergence Speed for Lightweight Neural Networks" by Zhao et al. (2021).

        https://arxiv.org/abs/2104.02639

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Scaling parameter (default 0.25).

    Returns
    -------
    torch.Tensor
        The element-wise TanhLinearUnit of the input.

    Example
    -------
    >>> import torch
    >>> from activations_plus.simple import tanh_linear_unit
    >>> x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> tanh_linear_unit(x)
    tensor([-0.9640, -0.7616,  0.0000,  1.0000,  2.0000])

    .. plot:: ../../examples/tanh_variants/tanh_linear_unit_example.py
       :include-source:

    """
    return torch.where(x >= 0, x, torch.tanh(x))


def stanhplus(x: Tensor, a: float = 1.0, b: float = 1.0, alpha: float = 1.0) -> Tensor:
    r"""Apply the Scaled TanhPlus activation function.

    .. math::

        \text{STanhPlus}(x) = a \tanh(bx) + \alpha x

    .. seealso::
        A variant of tanh activation with a learnable scale, discussed in "TanhPlus:
        A Modified Activation Function with Less Computation Cost" by Yan et al. (2020).

        https://arxiv.org/abs/2011.00055

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Scaling parameter for tanh (default 1.0).
    b : float, optional
        Scaling parameter inside tanh (default 1.0).
    alpha : float, optional
        Scaling parameter for linear term (default 1.0).

    Returns
    -------
    torch.Tensor
        The element-wise Scaled TanhPlus of the input.

    Example
    -------
    >>> import torch
    >>> from activations_plus.simple import stanhplus
    >>> x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> stanhplus(x, a=0.5, b=1.0, alpha=0.5)
    tensor([-1.4820, -0.8808, -0.0000,  0.8808,  1.4820])

    .. plot:: ../../examples/tanh_variants/stanhplus_example.py
       :include-source:

    """
    return a * torch.tanh(b * x) + alpha * x


def tanhsig(x: Tensor) -> Tensor:
    r"""Apply the TanhSig activation function.

    .. math::

        \text{TanhSig}(x) = x \cdot \tanh\left(\frac{2\pi}{3} \sigma(x)\right)

    .. seealso::
        Introduced in "TanhSig: A hybrid activation function" by Prajapati et al. (2021).

        Where \sigma(x) is the sigmoid function.

        https://doi.org/10.1016/j.neucom.2021.02.035

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise TanhSig of the input.

    Example
    -------
    >>> import torch
    >>> from activations_plus.simple import tanhsig
    >>> x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> tanhsig(x)
    tensor([-0.0952, -0.1517,  0.0000,  0.5644,  1.4576])

    .. plot:: ../../examples/tanh_variants/tanhsig_example.py
       :include-source:

    """
    return x * torch.tanh((2 * torch.pi / 3) * torch.sigmoid(x))
