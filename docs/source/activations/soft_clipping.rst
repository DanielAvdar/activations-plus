Soft Clipping
=============

.. automodule:: activations_plus.SoftClipping
   :members: forward, __init__


**Reference Paper**: [Soft Clipping Activation Function](https://arxiv.org/abs/2406.16640)

**Mathematical Explanation**:

The Soft Clipping activation function is defined as:

.. math::
    f(x) = \frac{x}{1 + |x|}

This ensures smooth clipping of input values.

Example Usage
-------------

.. code-block:: python

    import torch
    from activations_plus.soft_clipping import SoftClipping

    activation = SoftClipping(min_val=-2.0, max_val=2.0)
    input_tensor = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    output_tensor = activation(input_tensor)
    print(output_tensor)  # Example output: tensor([-1.9998, -1.2689,  0.0000,  1.2689,  1.9998])
