import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from activations_plus.simple import (
    relu_variants,
    sigmoid_tanh_variants,
    specialized_variants,
    tanh_variants,
)

# List of (function, kwargs) for parameterized testing
SIMPLE_ACTIVATIONS = [
    # Original activation functions
    (relu_variants.dual_line, {}),
    (sigmoid_tanh_variants.tanh_exp, {}),
    # (log_exp_softplus_variants.soft_exponential, {"a": -1}),
    # New ELU variants
    # New GELU/Swish variants
    # New Tanh variants
    (tanh_variants.penalized_tanh, {}),
    # Specialized variants (excluding prelu which requires weight parameter)
    (specialized_variants.resp, {}),
    (specialized_variants.erf_act, {}),
    (specialized_variants.hat, {}),
]


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
