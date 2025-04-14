SReLU
=====

.. automodule:: activations_plus.srelu.srelu_func
   :members:
   :undoc-members:
   :show-inheritance:

**Reference Paper**: [SReLU Activation Function](https://arxiv.org/abs/1512.07030)

**Example Usage**:

.. code-block:: python

    import torch
    from activations_plus.srelu import SReLU

    srelu = SReLU()
    input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    output_tensor = srelu(input_tensor)
    print(output_tensor)  # Example output: tensor([-1.5, -0.5,  0.0,  1.0,  2.0])
