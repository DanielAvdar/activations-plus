import torch
from hypothesis import given, strategies as st

from activations_plus.entmax.entmax import Entmax


def tensor_from_list(data, shape):
    return torch.tensor(data, dtype=torch.float32, requires_grad=True).reshape(shape)


@given(
    random_data=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=2, max_size=100
    ),
    dim=st.integers(min_value=-1, max_value=0),
)
def test_entmax_forward_pb(random_data, dim):
    shape = (len(random_data),)
    x = tensor_from_list(random_data, shape)

    entmax = Entmax(dim=dim)
    result = entmax(x)

    assert result is not None, "Output cannot be None"
    assert result.shape == x.shape, "Output shape mismatch"
    assert torch.all(result >= 0), "Output must be non-negative"
    assert torch.allclose(result.sum(dim), torch.ones_like(result.sum(dim)), atol=1e-4), (
        "Output must sum to 1 along the specified dimension"
    )


@given(
    random_data=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=2, max_size=100
    ),
    dim=st.integers(min_value=-1, max_value=0),
)
def test_entmax_backward_pb(random_data, dim):
    shape = (len(random_data),)
    x = tensor_from_list(random_data, shape).clone().detach().requires_grad_(True)

    entmax = Entmax(dim=dim)
    result = entmax(x)

    # Backward pass
    result.sum().backward()

    assert x.grad is not None, "Gradient should not be None"
    assert x.grad.shape == x.shape, "Gradient shape mismatch"
    grad_sum = x.grad.sum(dim)
    assert torch.allclose(grad_sum, torch.zeros_like(grad_sum), atol=1e-4), (
        "Gradients must sum to zero along the specified dimension"
    )