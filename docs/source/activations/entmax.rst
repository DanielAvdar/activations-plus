.. _entmax:

Entmax
======

.. automodule:: activations_plus.entmax.entmax_func
   :members:
   :undoc-members:
   :show-inheritance:

**Reference Paper**: [Entmax Activation Function](https://arxiv.org/abs/1905.05702)

.. code-block:: python

    import torch
    from activations_plus.entmax import Entmax

    activation = Entmax(dim=-1)
    x = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    y = activation(x)
    print("Entmax Output:", y)
