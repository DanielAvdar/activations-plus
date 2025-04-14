# Activations Plus

Activations Plus is a Python package designed to provide a collection of advanced activation functions for machine learning and deep learning models. These activation functions are implemented to enhance the performance of neural networks by addressing specific challenges such as sparsity, non-linearity, and gradient flow.

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/activations-plus)](https://pypi.org/project/activations-plus/)
[![version](https://img.shields.io/pypi/v/activations-plus)](https://img.shields.io/pypi/v/activations-plus)
[![License](https://img.shields.io/:license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![OS](https://img.shields.io/badge/ubuntu-blue?logo=ubuntu)
![OS](https://img.shields.io/badge/win-blue?logo=windows)
![OS](https://img.shields.io/badge/mac-blue?logo=apple)
[![Tests](https://github.com/DanielAvdar/activations-plus/actions/workflows/ci.yml/badge.svg)](https://github.com/DanielAvdar/activations-plus/actions/workflows/ci.yml)
[![Code Checks](https://github.com/DanielAvdar/activations-plus/actions/workflows/code-checks.yml/badge.svg)](https://github.com/DanielAvdar/activations-plus/actions/workflows/code-checks.yml)
[![codecov](https://codecov.io/gh/DanielAvdar/activations-plus/graph/badge.svg?token=N0V9KANTG2)](https://codecov.io/gh/DanielAvdar/activations-plus)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Last Commit](https://img.shields.io/github/last-commit/DanielAvdar/activations-plus/main)

## Features

- **Bent Identity**: Smooth approximation of the identity function.
- **ELiSH (Exponential Linear Squared Hyperbolic)**: Combines exponential and linear properties.
- **Entmax**: Sparse activation function for probabilistic models.
- **HardSwish**: Efficient approximation of the Swish activation.
- **Maxout**: Learns piecewise linear functions.
- **Soft Clipping**: Smoothly clips values to a range.
- **Sparsemax**: Sparse alternative to softmax.
- **SReLU (S-shaped Rectified Linear Unit)**: Combines linear and non-linear properties.

## Installation

To install the package, use pip:

```bash
pip install activations-plus
```

## Usage

Import and use any activation function in your PyTorch models:

```python
import torch
from activations_plus.bent_identity import BentIdentity

activation = BentIdentity()
x = torch.tensor([1.0, -1.0, 0.0])
output = activation(x)
print(output)
```

## Documentation

Comprehensive documentation is available [here](https://github.com/DanielAvdar/activations-plus).

## Supported Activation Functions

1. **Bent Identity**: A smooth approximation of the identity function. [Reference Paper](https://arxiv.org/abs/1901.08649)
2. **ELiSH**: Combines exponential and linear properties for better gradient flow. [Reference Paper](https://arxiv.org/abs/1901.08649)
3. **Entmax**: Sparse activation function for probabilistic models. [Reference Paper](https://arxiv.org/abs/1905.05702)
4. **HardSwish**: Efficient approximation of the Swish activation. [Reference Paper](https://arxiv.org/abs/1905.02244)
5. **Maxout**: Learns piecewise linear functions for better expressiveness. [Reference Paper](https://arxiv.org/abs/1302.4389)
6. **Soft Clipping**: Smoothly clips values to a range to avoid extreme outputs. [Reference Paper](https://arxiv.org/abs/2002.11002)
7. **Sparsemax**: Sparse alternative to softmax for probabilistic outputs. [Reference Paper](https://arxiv.org/abs/1602.02068)
8. **SReLU**: Combines linear and non-linear properties for better flexibility. [Reference Paper](https://arxiv.org/abs/1512.07030)

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## Testing

To run the tests, use the following command:

```bash
pytest tests/
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the contributors and the open-source community for their support.
