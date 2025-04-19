import pytest
import torch
from hypothesis import given, strategies as st

from activations_plus.entmax.entmax_func import Entmax15Function


def test_entmax15_forward_valid_input():
    x = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], dtype=torch.float32)
    result = Entmax15Function.apply(x)
    assert result is not None
    assert result.shape == x.shape
    assert torch.all(result >= 0)  # Ensure non-negativity
    assert torch.allclose(result.sum(dim=-1), torch.ones(result.size(0)), atol=1e-4)  # Sum to one


def test_entmax15_forward_invalid_dim():
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    with pytest.raises(IndexError):
        Entmax15Function.apply(x, 3)


def test_entmax15_backward():
    x = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], dtype=torch.float32, requires_grad=True)
    result = Entmax15Function.apply(x)
    result.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_entmax15_forward_zero_input():
    x = torch.zeros((2, 3), dtype=torch.float32)
    result = Entmax15Function.apply(x)
    expected = torch.tensor([[0.3333, 0.3333, 0.3333], [0.3333, 0.3333, 0.3333]], dtype=torch.float32)
    assert torch.allclose(result, expected, atol=1e-4)


@pytest.mark.skip
def test_entmax15_forward_dim_handling():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    result = Entmax15Function.apply(x, 0)
    assert result.shape == x.shape
    assert torch.all(result >= 0)
    assert torch.allclose(result.sum(dim=0), torch.ones(result.size(1)), atol=1e-4)


@pytest.mark.parametrize(
    "x",
    [
        torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),  # ascending input
        torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float32),  # descending input
        torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32),  # uniform input
        torch.tensor([[1.0, -1.0, 0.0]], dtype=torch.float32),  # mixed input
        torch.tensor([[10.0, 10.0, 10.0]], dtype=torch.float32),  # large uniform values
        torch.zeros((1, 3), dtype=torch.float32),  # zero input
    ],
)
def test_entmax15_backward_parametrized(x):
    x = x.clone().detach().requires_grad_(True)
    result = Entmax15Function.apply(x)
    result.sum().backward()

    assert x.grad is not None, "Gradient should not be None"
    assert x.grad.shape == x.shape, "Gradient should have same shape as input"
    grad_sum = x.grad.sum(-1)
    assert torch.allclose(grad_sum, torch.zeros_like(grad_sum), atol=1e-4), (
        "Gradients should sum to approximately zero along the projection dimension"
    )


def test_entmax15_forward_large_tensor():
    x = torch.randn(1000, 1000, dtype=torch.float32)
    result = Entmax15Function.apply(x)
    assert result is not None
    assert result.shape == x.shape
    assert torch.all(result >= 0)
    assert torch.allclose(result.sum(-1), torch.ones(result.size(0)), atol=1e-4)


def test_entmax15_forward_extreme_values():
    x = torch.tensor([[1e10, -1e10, 0.0], [1e-10, -1e-10, 0.0]], dtype=torch.float32)
    result = Entmax15Function.apply(x)
    assert result is not None
    assert result.shape == x.shape
    assert torch.all(result >= 0)


@given(
    random_data=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=2, max_size=100
    ),
    dim=st.integers(min_value=-1, max_value=0),
)
def test_entmax15_randomized(random_data, dim):
    shape = (len(random_data),)
    x = torch.tensor(random_data, dtype=torch.float32).reshape(shape).requires_grad_(True)
    result = Entmax15Function.apply(x, dim)

    assert result is not None, "Output cannot be None"
    assert result.shape == x.shape, "Output shape mismatch"
    assert torch.all(result >= 0), "Output must be non-negative"
    assert torch.allclose(result.sum(dim), torch.ones_like(result.sum(dim)), atol=1e-4), (
        "Output must sum to 1 along the specified dimension"
    )

    # Backward pass
    result.sum().backward()
    assert x.grad is not None, "Gradient should not be None"
    assert x.grad.shape == x.shape, "Gradient shape mismatch"
    grad_sum = x.grad.sum(dim)
    assert torch.allclose(grad_sum, torch.zeros_like(grad_sum), atol=1e-4), (
        "Gradients must sum to zero along the specified dimension"
    )


def test_entmax_math():
    x = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], dtype=torch.float32)
    result = Entmax15Function.apply(x)
    # Validate that the result is a projection onto the simplex and sums to 1
    assert torch.all(result >= 0), "Entmax output contains negative values."
    assert torch.allclose(result.sum(dim=-1), torch.ones(result.size(0)), atol=1e-4), "Entmax output does not sum to 1."


@pytest.mark.parametrize(
    "x",
    [
        torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32),
        torch.tensor([[-1.0, -2.0, -3.0], [0.5, 0.5, 0.5]], dtype=torch.float32),
    ],
)
def test_entmax_math_param(x):
    result = Entmax15Function.apply(x)
    assert torch.all(result >= 0), "Entmax output contains negative values."
    assert torch.allclose(result.sum(dim=-1), torch.ones(result.size(0)), atol=1e-4), "Entmax output does not sum to 1."


def test_entmax15_idempotence():
    x = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], dtype=torch.float32)
    y1 = Entmax15Function.apply(x)
    y2 = Entmax15Function.apply(y1)
    assert torch.allclose(y1, y2, atol=1e-5), "Entmax should be idempotent (applying twice yields same result)"


@pytest.mark.parametrize(
    "device", [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
)
def test_entmax15_device_consistency(device):
    x = torch.randn(5, 3, dtype=torch.float32, device=device, requires_grad=True)
    y = Entmax15Function.apply(x)
    assert y.device == x.device
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.device == x.device


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_entmax15_dtype_consistency(dtype):
    x = torch.randn(5, 3, dtype=dtype, requires_grad=True)
    y = Entmax15Function.apply(x)
    assert y.dtype == x.dtype


@pytest.mark.parametrize("shape", [(0, 3), (1, 3), (3, 0)])
def test_entmax15_empty_and_singleton(shape):
    x = torch.empty(*shape, dtype=torch.float32, requires_grad=True)
    y = Entmax15Function.apply(x)
    assert y.shape == x.shape


@pytest.mark.parametrize(
    "x",
    [
        torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32),
        torch.tensor([[-1.0, -2.0, -3.0], [0.5, 0.5, 0.5]], dtype=torch.float32),
    ],
)
def test_entmax15_monotonicity(x):
    # Monotonicity: increasing input should not decrease output
    x2 = x + 1.0
    y1 = Entmax15Function.apply(x)
    y2 = Entmax15Function.apply(x2)
    assert torch.all(y2 >= y1 - 1e-5), "Entmax output should be monotonic with respect to input"
