import hypothesis.extra.numpy as hnp
import torch
from hypothesis import given, strategies as st

from activations_plus.entmax.entmax import Entmax

from .entmax_tabnet import Entmax15 as EntmaxTabnet


# Property-based test for forward correctness (extended to higher dims)
@given(
    random_data=hnp.arrays(
        dtype=float,
        shape=hnp.array_shapes(min_dims=1, max_dims=4, min_side=2, max_side=10),
        elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    ),
    dim=st.integers(min_value=-1, max_value=0),
)
def test_entmax_forward_pb(random_data, dim):
    x = torch.tensor(random_data, dtype=torch.float64, requires_grad=True)
    entmax = Entmax(dim=dim)
    result = entmax(x)
    assert result is not None, "Output cannot be None"
    assert result.shape == x.shape, "Output shape mismatch"
    assert torch.all(result >= 0), "Output must be non-negative"
    assert torch.all(torch.isfinite(result)), "Output contains NaN or Inf"
    sum_result = result.sum(dim)
    assert torch.allclose(sum_result, torch.ones_like(sum_result), atol=1e-4), (
        "Output must sum to 1 along the specified dimension"
    )


# Property-based test for backward correctness (gradients)
@given(
    random_data=hnp.arrays(
        dtype=float,
        shape=hnp.array_shapes(min_dims=1, max_dims=4, min_side=2, max_side=10),
        elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    ),
    dim=st.integers(min_value=-1, max_value=0),
)
def test_entmax_backward_pb(random_data, dim):
    x = torch.tensor(random_data, dtype=torch.float64, requires_grad=True).clone().detach().requires_grad_(True)
    entmax = Entmax(dim=dim)
    result = entmax(x)
    result.sum().backward()
    assert x.grad is not None, "Gradient should not be None"
    assert x.grad.shape == x.shape, "Gradient shape mismatch"
    grad_sum = x.grad.sum(dim)
    assert torch.allclose(grad_sum, torch.zeros_like(grad_sum), atol=1e-4), (
        "Gradients must sum to zero along the specified dimension"
    )
    assert torch.all(torch.isfinite(x.grad)), "Gradient contains NaN or Inf"


# Comparison with TabNet Entmax reference implementation
@given(
    random_data=hnp.arrays(
        dtype=float,
        shape=hnp.array_shapes(min_dims=1, max_dims=4, min_side=2, max_side=10),
        elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    ),
)
def test_compare_with_tabnet_entmax(random_data):
    x = torch.tensor(random_data, dtype=torch.float64, requires_grad=True)
    for dim in range(-1, x.dim()):
        entmax = Entmax(dim=dim)
        entmax_tabnet = EntmaxTabnet(dim=dim)
        result = entmax(x)
        x_clone = x.clone().detach().requires_grad_(True)
        result_tabnet = entmax_tabnet(x_clone)
        assert torch.allclose(result, result_tabnet, atol=1e-5), (
            f"Entmax results do not match TabNet for dim={dim}: {result} vs {result_tabnet}"
        )
        result.sum().backward()
        result_tabnet.sum().backward()
        assert torch.allclose(x.grad, x_clone.grad, atol=1e-5), (
            f"Entmax gradients do not match TabNet for dim={dim}: {x.grad} vs {x_clone.grad}"
        )
