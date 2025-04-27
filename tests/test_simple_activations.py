import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from activations_plus.simple import (
    log_exp_softplus_variants,
    polynomial_power_variants,
    relu_variants,
    sigmoid_tanh_variants,
)

# List of (function, kwargs) for parameterized testing
SIMPLE_ACTIVATIONS = [
    (relu_variants.relu, {}),
    (relu_variants.lrelu, {}),
    (relu_variants.blrelu, {}),
    (relu_variants.rrelu, {}),
    (relu_variants.trec, {}),
    (relu_variants.dual_line, {}),
    (relu_variants.mrelu, {}),
    (sigmoid_tanh_variants.sigmoid, {}),
    (sigmoid_tanh_variants.tanh, {}),
    (sigmoid_tanh_variants.hardtanh, {}),
    (sigmoid_tanh_variants.softsign, {}),
    (sigmoid_tanh_variants.sqnl, {}),
    (sigmoid_tanh_variants.softplus, {}),
    (sigmoid_tanh_variants.tanh_exp, {}),
    (polynomial_power_variants.polynomial_linear_unit, {}),
    (polynomial_power_variants.power_function_linear_unit, {}),
    (polynomial_power_variants.power_linear_unit, {}),
    (polynomial_power_variants.inverse_polynomial_linear_unit, {}),
    (log_exp_softplus_variants.loglog, {}),
    (log_exp_softplus_variants.loglogish, {}),
    (log_exp_softplus_variants.logish, {}),
    (log_exp_softplus_variants.soft_exponential, {}),
    (log_exp_softplus_variants.softplus_linear_unit, {}),
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

    assert gradgradcheck(wrapped, (x,), eps=1e-6, atol=1e-4)
