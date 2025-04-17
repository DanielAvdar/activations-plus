Bent Identity
=============

.. automodule:: activations_plus.BentIdentity
   :members: forward

**Reference Paper**:
`Bent Identity Activation Function <https://arxiv.org/abs/1901.08649>`_

**Mathematical Explanation**:

The Bent Identity activation function is defined as:

.. math::
    f(x) = \frac{\sqrt{x^2 + 1} - 1}{2} + x

This introduces a slight non-linearity for negative inputs.

**Example Usage**:

.. code-block:: python

    import torch
    from activations_plus.bent_identity import BentIdentity

    activation = BentIdentity()
    x = torch.tensor([-3.0, 0.0, 3.0])
    y = activation(x)
    print("Bent Identity Output:", y)
