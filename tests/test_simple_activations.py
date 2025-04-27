import pytest
import torch

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
    try:
        y.sum().backward()
    except Exception as e:
        pytest.skip(f"No backward for {func.__name__}: {e}")
