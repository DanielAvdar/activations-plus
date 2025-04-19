import torch
from hypothesis import given, strategies as st
import hypothesis as hp
from activations_plus import Sparsemax


# Helper function to create tensors from hypothesis-generated lists
def tensor_from_list(data, shape):
    tensor = torch.tensor(data, dtype=torch.double, requires_grad=True).reshape(shape)
    return tensor


# Hypothesis test for forward correctness
@given(
    st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False), min_size=2, max_size=100
    ),
    st.integers(min_value=-1, max_value=0),
)
def test_sparsemax_forward_pb(random_data, dim):
    shape = (len(random_data),)
    x = tensor_from_list(random_data, shape)

    sparsemax = Sparsemax(dim=dim)
    result = sparsemax(x)

    assert result is not None, "Sparsemax forward output cannot be None"
    assert result.shape == x.shape, "Output shape must match input shape"
    assert torch.all(result >= 0), "Sparsemax output must have non-negative values"
    sum_result = result.sum(dim)
    assert torch.allclose(sum_result, torch.ones_like(sum_result)), (
        "Sparsemax outputs must sum to 1 along specified dimension or 0 (if sparsely activated)"
    )



# Hypothesis test for backward correctness (gradients)
@given(
    random_data=st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False), min_size=2, max_size=100
    ),
    dim=st.integers(min_value=-1, max_value=0),
)
def test_sparsemax_backward_pb(random_data, dim):
    shape = (len(random_data),)
    x = tensor_from_list(random_data, shape).clone().detach().requires_grad_(True)

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
