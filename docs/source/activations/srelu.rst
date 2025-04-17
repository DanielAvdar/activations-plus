SReLU
=====

.. automodule:: activations_plus.SReLU
   :members: forward, __init__

**Reference Paper**:
`SReLU Activation Function <https://arxiv.org/abs/1512.07030>`_

**Mathematical Explanation**:

The SReLU activation function is defined as:

.. math::
    f(x) = \begin{cases}
    t_1 + a_1(x - t_1), & x < t_1 \\
    x, & t_1 \leq x \leq t_2 \\
    t_2 + a_2(x - t_2), & x > t_2
    \end{cases}

where :math:`t_1`, :math:`t_2`, :math:`a_1`, and :math:`a_2` are learnable parameters.

**Example Usage**:

.. code-block:: python

    import torch
    from activations_plus.srelu import SReLU

    srelu = SReLU()
    input_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    output_tensor = srelu(input_tensor)
    print(output_tensor)  # Example output: tensor([-1.5, -0.5,  0.0,  1.0,  2.0])
