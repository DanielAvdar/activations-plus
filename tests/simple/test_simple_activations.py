import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from activations_plus.simple import (
    elu_variants,
    gelu_swish_variants,
    log_exp_softplus_variants,
    polynomial_power_variants,
    relu_variants,
    sigmoid_tanh_variants,
    sigmoid_variants,
    specialized_variants,
    tanh_variants,
)

# List of (function, kwargs) for parameterized testing
SIMPLE_ACTIVATIONS = [
    # Original activation functions
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
    # New ELU variants
    (elu_variants.elu, {}),
    (elu_variants.selu, {}),
    (elu_variants.celu, {}),
    (elu_variants.abslu, {}),
    # New GELU/Swish variants
    (gelu_swish_variants.gelu, {}),
    (gelu_swish_variants.silu, {}),
    (gelu_swish_variants.swish, {}),
    (gelu_swish_variants.hard_sigmoid, {}),
    (gelu_swish_variants.hard_swish, {}),
    (gelu_swish_variants.mish, {}),
    (gelu_swish_variants.phish, {}),
    # New Tanh variants
    (tanh_variants.tanh_linear_unit, {}),
    (tanh_variants.penalized_tanh, {}),
    (tanh_variants.stanhplus, {}),
    (tanh_variants.tanhsig, {}),
    # New Sigmoid variants
    (sigmoid_variants.rootsig, {}),
    (sigmoid_variants.new_sigmoid, {}),
    (sigmoid_variants.sigmoid_gumbel, {}),
    (sigmoid_variants.root2sigmoid, {}),
    # Specialized variants (excluding prelu which requires weight parameter)
    (specialized_variants.resp, {}),
    (specialized_variants.suish, {}),
    (specialized_variants.sin_sig, {}),
    (specialized_variants.gish, {}),
    (specialized_variants.erf_act, {}),
    (specialized_variants.complementary_log_log, {}),
    (specialized_variants.exp_expish, {}),
    (specialized_variants.exp_swish, {}),
    (specialized_variants.hat, {}),
]

# Functions that don't support double backward
SKIP_DOUBLE_BACKWARD = [
    gelu_swish_variants.hard_sigmoid,
    gelu_swish_variants.hard_swish,  # This may also have issues with second derivatives
]


# Special case for prelu which requires weight parameter
def test_prelu():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    weight = torch.tensor([0.1], requires_grad=True)
    y = specialized_variants.prelu(x, weight)
    assert torch.is_tensor(y)
    assert y.shape == x.shape
    # Check backward
    y.sum().backward()
    assert x.grad is not None
    assert weight.grad is not None


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
    if func in SKIP_DOUBLE_BACKWARD:
        pytest.skip(f"Skipping double backward test for {func.__name__} as it's not supported")

    assert gradgradcheck(wrapped, (x,), eps=1e-6, atol=1e-4)
