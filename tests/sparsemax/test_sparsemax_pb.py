import hypothesis.extra.numpy as hnp
import torch
from hypothesis import given, strategies as st

from activations_plus import Sparsemax
from activations_plus.sparsemax.sparsemax_func_v2 import SparsemaxFunction

from .tabnet_original import Sparsemax as SparsemaxOriginal


# Helper function to create tensors from hypothesis-generated lists
def tensor_from_list(data, shape):
    tensor = torch.tensor(data, dtype=torch.double, requires_grad=True).reshape(shape)
    return tensor


# Hypothesis test for forward correctness
@given(
    random_data=hnp.arrays(
        dtype=float,
        shape=hnp.array_shapes(min_dims=2, max_dims=5, min_side=2, max_side=10),
        elements=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
    ),
    dim=st.integers(min_value=-1, max_value=0),
)
def test_sparsemax_forward_pb(random_data, dim):
    x = torch.tensor(random_data, dtype=torch.double, requires_grad=True)

    sparsemax = Sparsemax(dim=dim)
    result = sparsemax(x)

    assert result is not None, "Sparsemax forward output cannot be None"
    assert result.shape == x.shape, "Output shape must match input shape"
    assert torch.all(result >= 0), "Sparsemax output must have non-negative values"
    # General assertions for more thorough testing
    assert torch.all(torch.isfinite(result)), "Sparsemax output contains NaN or Inf"
    sum_result = result.sum(dim)
    assert torch.allclose(sum_result, torch.ones_like(sum_result)), (
        "Sparsemax outputs must sum to 1 along specified dimension or 0 (if sparsely activated)"
    )


# Hypothesis test for backward correctness (gradients)
@given(
    random_data=hnp.arrays(
        dtype=float,
        shape=hnp.array_shapes(min_dims=1, max_dims=5, min_side=2, max_side=5),
        elements=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
    ),
    dim=st.integers(min_value=-1, max_value=0),
)
def test_sparsemax_backward_pb(random_data, dim):
    x = torch.tensor(random_data, dtype=torch.double, requires_grad=True).clone().detach().requires_grad_(True)

    # Apply sparsemax and conduct backward pass
    sparsemax = Sparsemax(dim=dim)
    result = sparsemax(x)

    # Perform backward pass with a sum reduction
    result.sum().backward()

    assert x.grad is not None, "Gradient should not be None"
    assert x.grad.shape == x.shape, "Gradient shape must match input shape"

    # Check gradient property: sparsemax gradient sum over each projection dim should be zero
    grads_sum = x.grad.sum(dim)
    zeros_tensor = torch.zeros_like(grads_sum)

    assert torch.allclose(grads_sum, zeros_tensor, atol=1e-5), (
        "Gradients must sum to zero along the sparsemax dimension"
    )

    # Check gradient is zero where output is zero
    output = sparsemax(x)
    assert torch.all(x.grad[output == 0] == 0), "Gradient should be zero where output is zero"
    # General assertions for more thorough testing
    assert torch.all(torch.isfinite(x.grad)), "Gradient contains NaN or Inf"
    # Check all gradients are finite
    assert torch.all(torch.isfinite(x.grad)), "All gradients should be finite (no NaN or Inf)"


# Hypothesis test for Sparsemax v2 threshold and support values correctness
@given(
    random_data=hnp.arrays(
        dtype=float,
        shape=hnp.array_shapes(min_dims=1, max_dims=5, min_side=2, max_side=5),
        elements=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
    ),
    dim=st.integers(min_value=-1, max_value=0),
)
def test_sparsemax_v2_threshold_and_support(random_data, dim):
    x = torch.tensor(random_data, dtype=torch.double)
    tau, support_size = SparsemaxFunction._threshold_and_support(x, dim=dim)
    support_mask = (x - tau) > 0
    expected_support_size = support_mask.sum(dim=dim, keepdim=True)
    assert torch.all(support_size == expected_support_size), "Support size does not match mask count"
    sum_positive = (x - tau).clamp(min=0).sum(dim=dim, keepdim=True)
    # The sum of positive parts should be 1 for each row (sparsemax property)
    assert torch.allclose(sum_positive, torch.ones_like(sum_positive)), (
        f"Sum of positive parts is not 1: got {sum_positive}"
    )


@given(
    random_data=hnp.arrays(
        dtype=float,
        shape=hnp.array_shapes(min_dims=1, max_dims=5, min_side=2, max_side=5),
        elements=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
    ),
)
def test_compare_with_original(random_data):
    x = torch.tensor(random_data, dtype=torch.double)
    for dim in range(-1, x.dim()):
        sparsemax_v2 = Sparsemax(dim=dim)
        sparsemax_original = SparsemaxOriginal(dim=dim)

        result_v2 = sparsemax_v2(x)
        result_original = sparsemax_original(x)

        assert torch.allclose(result_v2, result_original, atol=1e-5), (
            f"Results do not match for dim={dim}: {result_v2} vs {result_original}"
        )
