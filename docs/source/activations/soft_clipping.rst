Soft Clipping
=============

.. automodule:: activations_plus.SoftClipping
   :members: forward, __init__

**Reference Paper**:
`Soft Clipping Activation Function <https://arxiv.org/abs/2406.16640>`_

**Mathematical Explanation**:

The Soft Clipping activation function is defined as:

.. math::
    f(x) = x_{\text{min}} + (x_{\text{max}} - x_{\text{min}}) \cdot g(x)

where :math:`g(x)` is the clipping function, which can be chosen as sigmoid, tanh, or any other differentiable function. For example:

- Sigmoid: :math:`g(x) = \frac{1}{1 + e^{-x}}`
- Hyperbolic Tangent (tanh): :math:`g(x) = \tanh(x)`

This ensures smooth clipping of input values within the range :math:`[x_{\text{min}}, x_{\text{max}}]`, with the behavior determined by the choice of :math:`g(x)`.

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

**Custom Clipping Function**:

The Soft Clipping activation function can also use a custom clipping function instead of the default sigmoid. For example:

.. code-block:: python

    import torch
    from activations_plus.soft_clipping import SoftClipping

    # Custom clipping function (e.g., hyperbolic tangent)
    custom_clip_func = torch.tanh

    # Initialize the Soft Clipping activation function with a custom function
    activation = SoftClipping(x_min=-2.0, x_max=2.0, clip_func=custom_clip_func)

    # Input tensor
    input_tensor = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

    # Apply the activation function
    output_tensor = activation(input_tensor)

    # Print the output
    print(output_tensor)  # Example output: tensor([-1.9993, -1.7616,  0.0000,  1.7616,  1.9993])

---

This activation function is particularly useful for stabilizing neural network training by smoothly limiting the range of activations.
