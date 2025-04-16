Sparsemax
=========

.. automodule:: activations_plus.Sparsemax
   :members: forward, __init__


**Reference Paper**: [Sparsemax Activation Function](https://arxiv.org/abs/1602.02068)

**Mathematical Explanation**:

The Sparsemax activation function maps inputs to a probability distribution, similar to softmax, but encourages sparsity by projecting onto a simplex.

The Sparsemax activation function is defined as:

.. math::
    \text{Sparsemax}(z) = \underset{p \in \Delta^{d-1}}{\operatorname{argmin}} \|p - z\|^2

where :math:`\Delta^{d-1}` is the :math:`(d-1)`-dimensional probability simplex.

**Example Usage**:

.. code-block:: python

    import torch
    from activations_plus.sparsemax import SparsemaxFunction
    input_tensor = torch.tensor([[0.5, 2.0, 1.0], [1.0, 0.0, 3.0]])
    sparsemax = SparsemaxFunction.apply
    output_tensor = sparsemax(input_tensor, 1)  # Pass the dimension as a positional argument
    print(output_tensor)  # Example output: tensor([[0.0000, 0.6667,
