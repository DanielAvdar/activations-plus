"""ReLU and Leaky ReLU variants for PyTorch.

This module provides several simple ReLU-based activation functions.
"""

import torch
from torch import Tensor


def relu(x: Tensor) -> Tensor:
    r"""Apply the Rectified Linear Unit activation.

    .. math::

        \mathrm{ReLU}(z) = \max(0, z)


    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise ReLU of the input.


    Source
    ------
    .. seealso::
        First proposed in "Rectified Linear Units Improve Restricted Boltzmann Machines"
        by Nair & Hinton (2010).

        https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf

    Example
    -------
    .. plot:: ../../examples/relu_variants/relu_example.py
       :include-source:

    """
    return torch.relu(x)


def lrelu(x: Tensor, a: float = 0.01) -> Tensor:
    r"""Apply the Leaky ReLU activation.

    .. math::

        \mathrm{LReLU}(z) = \begin{cases} z, & z \geq 0 \\ \frac{z}{a}, & z < 0 \end{cases}

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


    Source
    ------
    .. seealso::
        Introduced in "Rectifier Nonlinearities Improve Neural Network Acoustic Models"
        by Maas et al. (2013).

        https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf

    Example
    -------


    .. plot:: ../../examples/relu_variants/lrelu_example.py
       :include-source:

    """
    return torch.where(x >= 0, x, x / a)


def blrelu(x: Tensor, a: float = 0.01, b: float = 1.0, c: float = 0.0) -> Tensor:
    r"""Apply the Bounded Leaky ReLU activation.

    .. math::

        \mathrm{BLReLU}(z) = \begin{cases} az, & z \leq 0 \\ z, & 0 < z < b \\ az + c, & z \geq b \end{cases}

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


    Source
    ------
    .. seealso::
        Introduced in "Activation Functions in Neural Networks: A Systematic Overview"
        by Dubey et al. (2022).

        https://arxiv.org/abs/2110.09084

    Example
    -------


    .. plot:: ../../examples/relu_variants/blrelu_example.py
       :include-source:

    """
    return torch.where(x <= 0, a * x, torch.where((x > 0) & (x < b), x, a * x + c))


def rrelu(x: Tensor, a: float = 0.01) -> Tensor:
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


    Source
    ------
    .. seealso::
        Proposed in "Empirical Evaluation of Rectified Activations in Convolutional Network"
        by Xu et al. (2015).

        https://arxiv.org/abs/1505.00853

    Example
    -------


    .. plot:: ../../examples/relu_variants/rrelu_example.py
       :include-source:

    """
    return torch.where(x >= 0, x, x * a)


def trec(x: Tensor, a: float = 0.0) -> Tensor:
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


    Source
    ------
    .. seealso::
        A variant of ReLU with an adjustable threshold, discussed in "Neural Networks with Piecewise
        Activation Functions" by Zhao & Li (2020).

        https://arxiv.org/abs/2003.01491

    Example
    -------

    .. plot:: ../../examples/relu_variants/trec_example.py
       :include-source:

    """
    return torch.where(x > a, x, torch.zeros_like(x))


def dual_line(x: Tensor, a: float = 1.0, b: float = 0.01, m: float = 0.0) -> Tensor:
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


    Source
    ------
    .. seealso::
        A generalized linear activation function discussed in "Survey of Activation Functions for Deep Neural
        Networks" by Nwankpa et al. (2018).

        https://arxiv.org/abs/1811.03378

    Example
    -------


    .. plot:: ../../examples/relu_variants/dual_line_example.py
       :include-source:

    """
    return torch.where(x >= 0, a * x + m, b * x + m)


def mrelu(x: Tensor) -> Tensor:
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


    Source
    ------
    .. seealso::
        A variant of ReLU discussed in "On Advanced Activation Functions for Deep Learning"
        by Chen et al. (2020).

        https://arxiv.org/abs/2011.05627

    Example
    -------


    .. plot:: ../../examples/relu_variants/mrelu_example.py
       :include-source:

    """
    return torch.min(torch.relu(1 - x), torch.relu(1 + x))
