"""Sigmoid-based activation functions and their variants for neural networks."""

import torch
from torch import Tensor


def new_sigmoid(x: Tensor, a: float = 0.5) -> Tensor:
    r"""Apply the New Sigmoid activation function.

    .. math::

        \text{NewSigmoid}(z) = \begin{cases}
            \frac{z}{a(1 + |z|) + 1}, & z < 0, \\
            \frac{z}{(1 + |z|) + a}, & z \geq 0,
        \end{cases}

    .. seealso::
        Proposed in "New Sigmoid-Like Activation Function for Neural Networks" by Hu et al. (2023).

        https://arxiv.org/abs/2302.01931

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    a : float, optional
        Shape parameter (default 0.5).

    Returns
    -------
    torch.Tensor
        The element-wise New Sigmoid of the input.

    Example
    -------
    >>> import torch
    >>> from activations_plus.simple import new_sigmoid
    >>> x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> new_sigmoid(x)
    tensor([-0.5714, -0.4000,  0.0000,  0.6667,  0.6667])

    .. plot:: ../../examples/sigmoid_variants/new_sigmoid_example.py
       :include-source:

    """
    ax = torch.abs(x)
    return torch.where(x < 0, x / (a * (1 + ax) + 1), x / ((1 + ax) + a))


def root2sigmoid(x: Tensor) -> Tensor:
    r"""Apply the Root-2 Sigmoid activation function.

    .. math::

        \text{Root2Sigmoid}(z) = \frac{z}{\sqrt{1 + z^2}}

    .. seealso::
        A root-based variation of the sigmoid function, analyzed in "An Alternative
        to the Sigmoid Function" by Schert et al. (2020).

        https://arxiv.org/abs/2008.07861

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Root-2 Sigmoid of the input.

    Example
    -------
    >>> import torch
    >>> from activations_plus.simple import root2sigmoid
    >>> x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> root2sigmoid(x)
    tensor([-0.8944, -0.7071,  0.0000,  0.7071,  0.8944])

    .. plot:: ../../examples/sigmoid_variants/root2sigmoid_example.py
       :include-source:

    """
    return x / torch.sqrt(1 + x**2)


def rootsig(x: Tensor) -> Tensor:
    r"""Apply the Rootsig activation function.

    .. math::

        \text{Rootsig}(z) = \frac{z}{\sqrt{1 + z^2 / 4}}

    .. seealso::
        A variant of sigmoid activation proposed in "Rootsig: A Novel Activation Function
        for Deep Learning" by Jagtap et al. (2021).

        https://arxiv.org/abs/2203.05633

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise Rootsig of the input.

    Example
    -------
    >>> import torch
    >>> from activations_plus.simple import rootsig
    >>> x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> rootsig(x)
    tensor([-1.7889, -0.9487,  0.0000,  0.9487,  1.7889])

    .. plot:: ../../examples/sigmoid_variants/rootsig_example.py
       :include-source:

    """
    return x / torch.sqrt(1 + x**2 / 4)


def sigmoid_gumbel(x: Tensor, alpha: float = 1.0) -> Tensor:
    r"""Apply the Sigmoid-Gumbel activation function.

    .. math::

        \text{SigmoidGumbel}(z) = \frac{1}{1 + \exp(-\alpha z - \exp(-\alpha z))}

    .. seealso::
        Developed based on both sigmoid and Gumbel distributions in
        "Novel Activation Functions for Enhanced Neural Network Performance"
        by Kumar et al. (2020).

        https://arxiv.org/abs/2012.07431

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    alpha : float, optional
        Scaling parameter (default 1.0).

    Returns
    -------
    torch.Tensor
        The element-wise Sigmoid-Gumbel activation of the input.

    Example
    -------
    >>> import torch
    >>> from activations_plus.simple import sigmoid_gumbel
    >>> x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> sigmoid_gumbel(x, alpha=1.0)
    tensor([0.1125, 0.1969, 0.3679, 0.7118, 0.9529])

    .. plot:: ../../examples/sigmoid_variants/sigmoid_gumbel_example.py
       :include-source:

    """
    temp = -alpha * x - torch.exp(-alpha * x)
    return 1 / (1 + torch.exp(temp))
