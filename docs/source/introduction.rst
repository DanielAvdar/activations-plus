.. _introduction:

Activations Plus
================

Activations Plus is a Python package designed to provide a collection of advanced activation functions for machine learning and deep learning models. These activation functions are implemented to enhance the performance of neural networks by addressing specific challenges such as sparsity, non-linearity, and gradient flow.

The package includes a variety of activation functions, such as:

- Bent Identity (Note: Experimental)
- ELiSH (Exponential Linear Squared Hyperbolic) (Note: Experimental)
- Entmax: A flexible sparse activation function for probabilistic models.
- HardSwish (Note: Experimental)
- Maxout (Note: Experimental)
- Soft Clipping (Note: Experimental)
- Sparsemax: A sparse alternative to softmax, useful for probabilistic outputs.
- SReLU (S-shaped Rectified Linear Unit) (Note: Experimental)

Each activation function is implemented with efficiency and flexibility in mind, making it easy to integrate into existing machine learning pipelines. Whether you're working on classification, regression, or other tasks, Activations Plus provides tools to experiment with and optimize your models.

Below are examples of Sparsemax and Entmax in action:

```python
import torch
from activations_plus.sparsemax import Sparsemax
from activations_plus.entmax import Entmax

# Sparsemax Example
sparsemax = Sparsemax()
x = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, -1.0]])
output_sparsemax = sparsemax(x)
print("Sparsemax Output:", output_sparsemax)

# Entmax Example
entmax = Entmax(alpha=1.5)
output_entmax = entmax(x)
print("Entmax Output:", output_entmax)
```

These examples illustrate how to use Sparsemax and Entmax activation functions in PyTorch.
