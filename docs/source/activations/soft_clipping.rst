Soft Clipping
=============

.. automodule:: activations_plus.SoftClipping
   :members: forward, __init__

**Reference Paper**:
`Soft Clipping Activation Function <https://arxiv.org/abs/2406.16640>`_

**Mathematical Explanation**:

The Soft Clipping activation function is defined as:

.. math::
    f(x) = x_{\text{min}} + (x_{\text{max}} - x_{\text{min}}) \cdot \sigma(x)

where :math:`\sigma(x)` is the sigmoid function, defined as:

.. math::
    \sigma(x) = \frac{1}{1 + e^{-x}}

This ensures smooth clipping of input values within the range :math:`[x_{\text{min}}, x_{\text{max}}]`.

---

**Example Usage**:

.. code-block:: python

    import torch
    from activations_plus.soft_clipping import SoftClipping

    # Initialize the Soft Clipping activation function
    activation = SoftClipping(x_min=-2.0, x_max=2.0)

    # Input tensor
    input_tensor = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

    # Apply the activation function
    output_tensor = activation(input_tensor)

    # Print the output
    print(output_tensor)  # Example output: tensor([-1.9998, -1.2689,  0.0000,  1.2689,  1.9998])

---

This activation function is particularly useful for stabilizing neural network training by smoothly limiting the range of activations.
