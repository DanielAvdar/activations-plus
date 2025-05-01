import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

import activations_plus.simple as simple
from activations_plus.simple import (
    __all__ as all_simple,
)

all_simple_functions = [getattr(simple, func_name) for func_name in all_simple]

SIMPLE_ACTIVATIONS = [(func, {}) for func in all_simple_functions]


@pytest.mark.parametrize("func, kwargs", SIMPLE_ACTIVATIONS)
def test_simple_activations(func, kwargs):
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    # Some functions require extra args, so pass kwargs
    y = func(x, **kwargs)
    assert torch.is_tensor(y)
    assert y.shape == x.shape
    # Check backward if possible
    y.sum().backward()
    assert x.grad is not None


@pytest.mark.parametrize("func, kwargs", SIMPLE_ACTIVATIONS)
def test_simple_activations_gradcheck(func, kwargs):
    x = torch.randn(3, 3, dtype=torch.double, requires_grad=True)

    # gradcheck expects the function to return only doubles and take only doubles
    def wrapped(x):
        return func(x, **kwargs)

    assert gradcheck(wrapped, (x,), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("func, kwargs", SIMPLE_ACTIVATIONS)
def test_simple_activations_gradgradcheck(func, kwargs):
    x = torch.randn(3, 3, dtype=torch.double, requires_grad=True)

    def wrapped(x):
        return func(x, **kwargs)

    # Skip double backward tests for functions that don't support it

    assert gradgradcheck(wrapped, (x,), eps=1e-6, atol=1e-4)
