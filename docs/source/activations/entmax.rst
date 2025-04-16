.. _entmax:

Entmax
======

.. automodule:: activations_plus.Entmax
   :members: forward, __init__


**Reference Paper**: [Entmax Activation Function](https://arxiv.org/abs/1905.05702)

**Mathematical Explanation**:

The Entmax activation function is defined as:

.. math::
    \text{Entmax}_\alpha(z) = \underset{p \in \Delta^{d-1}}{\operatorname{argmax}} \left( p \cdot z - \frac{1}{\alpha(\alpha-1)} \sum_{i=1}^d p_i^\alpha \right)

where :math:`\alpha` controls the sparsity of the output.

.. code-block:: python

    import torch
    from activations_plus.entmax import Entmax

    activation = Entmax(dim=-1)
    x = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    y = activation(x)
    print("Entmax Output:", y)
