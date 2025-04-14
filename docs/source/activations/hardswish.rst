.. _hardswish:

Hardswish
=========

.. automodule:: activations_plus.HardSwish
   :members: forward, __init__


**Reference Paper**: [HardSwish Activation Function](https://arxiv.org/abs/1905.02244)

.. code-block:: python

    import torch
    from activations_plus.hardswish import HardSwish

    activation = HardSwish()
    x = torch.tensor([-3.0, 0.0, 3.0])
    y = activation(x)
    print("HardSwish Output:", y)
