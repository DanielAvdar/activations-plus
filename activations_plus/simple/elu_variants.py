"""ELU-based activation functions and their variants for neural networks."""

import torch
from torch import Tensor


def isrlu(x: Tensor, alpha: float = 1.0) -> Tensor:
    r"""Apply the Inverse Square Root Linear Unit activation function.

    .. math::

        \text{ISRLU}(x) = \begin{cases}
            x, & x \geq 0 \\
            \frac{x}{\sqrt{1 + \alpha x^2}}, & x < 0
        \end{cases}


    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    alpha : float, optional
        Scaling parameter (default 1.0).

    Returns
    -------
    torch.Tensor
        The element-wise ISRLU of the input.



    Source
    ------
    .. seealso::
        Proposed in **"Improving Deep Neural Networks with New Activation Functions"**
        by Carlile et al. (2017).

        `arxiv <https://arxiv.org/abs/1710.09967>`_

    Example
    -------

    .. plot:: ../../examples/elu_variants/isrlu_example.py
       :include-source:

    """
    return torch.where(x >= 0, x, x / torch.sqrt(1 + alpha * x.pow(2)))
