Bent Identity
=============

.. automodule:: activations_plus.bent_identity.bent_identity_func
   :members:
   :undoc-members:
   :show-inheritance:

**Reference Paper**: [Bent Identity Activation Function](https://arxiv.org/abs/1901.08649)

.. code-block:: python

    import torch
    from activations_plus.bent_identity import BentIdentity

    activation = BentIdentity()
    x = torch.tensor([-3.0, 0.0, 3.0])
    y = activation(x)
    print("Bent Identity Output:", y)
