.. _elish:

Elish
=====

.. automodule:: activations_plus.ELiSH
   :members: forward, __init__


**Reference Paper**: [ELiSH Activation Function](https://arxiv.org/abs/1808.00783)

.. code-block:: python

    import torch
    from activations_plus.elish import ELiSH

    activation = ELiSH()
    x = torch.tensor([-3.0, 0.0, 3.0])
    y = activation(x)
    print("ELiSH Output:", y)
