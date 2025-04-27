.. _examples:

Simple Activation Functions Examples
====================================

This page demonstrates usage examples for the simple activation functions provided in the ``activations_plus.simple`` subpackage.

.. code-block:: python

    import torch
    from activations_plus.simple import relu_variants, sigmoid_tanh_variants

    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    # ReLU variant
    y_relu = relu_variants.relu(x)
    print('ReLU:', y_relu)

    # Leaky ReLU
    y_lrelu = relu_variants.lrelu(x, a=0.1)
    print('Leaky ReLU:', y_lrelu)

    # Sigmoid
    y_sigmoid = sigmoid_tanh_variants.sigmoid(x)
    print('Sigmoid:', y_sigmoid)

    # HardTanh
    y_hardtanh = sigmoid_tanh_variants.hardtanh(x, a=-1.0, b=1.0)
    print('HardTanh:', y_hardtanh)

    # Softsign
    y_softsign = sigmoid_tanh_variants.softsign(x)
    print('Softsign:', y_softsign)

You can use any of the available functions in a similar way. See the API reference for the full list of functions and their parameters.
