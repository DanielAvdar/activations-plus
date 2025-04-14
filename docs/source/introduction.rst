.. _introduction:

Introduction
============

.. image:: https://img.shields.io/pypi/v/activations-plus.svg
   :target: https://pypi.org/project/activations-plus/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/activations-plus.svg
   :target: https://pypi.org/project/activations-plus/
   :alt: Python versions

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

.. image:: https://img.shields.io/badge/ubuntu-blue?logo=ubuntu
   :alt: Ubuntu

.. image:: https://img.shields.io/badge/ubuntu-blue?logo=windows
   :alt: Windows

.. image:: https://img.shields.io/badge/ubuntu-blue?logo=apple
   :alt: MacOS

.. image:: https://codecov.io/gh/DanielAvdar/activations-plus/graph/badge.svg?token=N0V9KANTG2
   :alt: Coverage

.. image:: https://img.shields.io/github/last-commit/DanielAvdar/activations-plus/main
   :alt: Last Commit



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

Installation
============
To install activations-plus, use pip:

.. code-block:: bash

    pip install activations-plus

Usage
=====
Below are examples of Sparsemax and Entmax in action:

.. code-block:: python

    import torch
    from activations_plus.sparsemax import Sparsemax
    from activations_plus.entmax import Entmax

    # Sparsemax Example
    sparsemax = Sparsemax()
    x = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, -1.0]])
    output_sparsemax = sparsemax(x)
    print("Sparsemax Output:", output_sparsemax)

    # Entmax Example
    entmax = Entmax(dim=-1)
    output_entmax = entmax(x)
    print("Entmax Output:", output_entmax)

These examples illustrate how to use Sparsemax and Entmax activation functions in PyTorch.
