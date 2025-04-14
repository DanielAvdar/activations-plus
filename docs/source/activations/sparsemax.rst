Sparsemax
=========

.. automodule:: activations_plus.Sparsemax
   :members: forward, __init__


**Reference Paper**: [Sparsemax Activation Function](https://arxiv.org/abs/1602.02068)

**Example Usage**:

.. code-block:: python

    import torch
    from activations_plus.sparsemax import SparsemaxFunction
    input_tensor = torch.tensor([[0.5, 2.0, 1.0], [1.0, 0.0, 3.0]])
    sparsemax = SparsemaxFunction.apply
    output_tensor = sparsemax(input_tensor, 1)  # Pass the dimension as a positional argument
    print(output_tensor)  # Example output: tensor([[0.0000, 0.6667,
