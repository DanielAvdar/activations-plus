.. _examples:

Simple Activation Functions Examples
====================================

This page demonstrates usage examples and visualizations for the simple activation functions provided in the ``activations_plus.simple`` subpackage.

.. plot::
   :include-source:

   import torch
   import numpy as np
   import matplotlib.pyplot as plt
   from activations_plus.simple import relu_variants, sigmoid_tanh_variants, polynomial_power_variants, log_exp_softplus_variants

   x = torch.linspace(-3, 3, 200)
   x_np = x.numpy()

   activations = [
       (relu_variants.relu, "ReLU"),
       (relu_variants.lrelu, "Leaky ReLU (a=0.1)", {"a": 0.1}),
       (relu_variants.blrelu, "Bounded Leaky ReLU", {}),
       (relu_variants.rrelu, "Randomized Leaky ReLU", {}),
       (relu_variants.trec, "Truncated ReLU", {}),
       (relu_variants.dual_line, "Dual Line", {}),
       (relu_variants.mrelu, "Mirrored ReLU", {}),
       (sigmoid_tanh_variants.sigmoid, "Sigmoid", {}),
       (sigmoid_tanh_variants.tanh, "Tanh", {}),
       (sigmoid_tanh_variants.hardtanh, "HardTanh", {}),
       (sigmoid_tanh_variants.softsign, "Softsign", {}),
       (sigmoid_tanh_variants.sqnl, "SQNL", {}),
       (sigmoid_tanh_variants.softplus, "Softplus", {}),
       (sigmoid_tanh_variants.tanh_exp, "TanhExp", {}),
       (polynomial_power_variants.polynomial_linear_unit, "Polynomial Linear Unit", {}),
       (polynomial_power_variants.power_function_linear_unit, "Power Function Linear Unit", {}),
       (polynomial_power_variants.power_linear_unit, "Power Linear Unit", {}),
       (polynomial_power_variants.inverse_polynomial_linear_unit, "Inverse Polynomial Linear Unit", {}),
       (log_exp_softplus_variants.loglog, "LogLog", {}),
       (log_exp_softplus_variants.loglogish, "LogLogish", {}),
       (log_exp_softplus_variants.logish, "Logish", {}),
       (log_exp_softplus_variants.soft_exponential, "Soft Exponential", {}),
       (log_exp_softplus_variants.softplus_linear_unit, "Softplus Linear Unit", {}),
   ]

   n = len(activations)
   ncols = 3
   nrows = int(np.ceil(n / ncols))
   fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3))
   axes = axes.flatten()
   for i, (func, name, *opt) in enumerate(activations):
       kwargs = opt[0] if opt else {}
       y = func(x, **kwargs).detach().numpy()
       ax = axes[i]
       ax.plot(x_np, y, color='#1f77b4', linewidth=2)
       ax.set_title(name)
       ax.set_xlabel('Input')
       ax.set_ylabel('Output')
       ax.grid(True, alpha=0.3)
   for j in range(i+1, len(axes)):
       fig.delaxes(axes[j])
   fig.tight_layout()
   plt.show()

You can use any of the available functions in a similar way. See the API reference for the full list of functions and their parameters.
