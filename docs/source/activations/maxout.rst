.. _maxout:

Maxout
======

.. automodule:: activations_plus.Maxout
   :members: forward, __init__


**Reference Paper**: [Maxout Activation Function](https://arxiv.org/abs/1302.4389)

.. code-block:: python

    import torch
    from activations_plus.maxout import Maxout

    activation = Maxout(num_pieces=2)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = activation(x)
    print("Maxout Output:", y)
