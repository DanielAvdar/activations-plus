import pytest
import torch

from activations_plus.sparsemax import SparsemaxFunction


def test_sparsemax_forward_valid_input():
    x = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], dtype=torch.float32)
    result = SparsemaxFunction.apply(x)
    assert result is not None
    assert result.shape == x.shape
    assert torch.all(result >= 0)


def test_sparsemax_forward_invalid_dim():
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    with pytest.raises(IndexError):
        SparsemaxFunction.apply(x, 3)


def test_sparsemax_backward():
    x = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], dtype=torch.float32, requires_grad=True)
    result = SparsemaxFunction.apply(x)
    result.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_sparsemax_forward_zero_input():
    x = torch.zeros((2, 3), dtype=torch.float32)
    result = SparsemaxFunction.apply(x)
    expected = torch.tensor([[0.3333, 0.3333, 0.3333], [0.3333, 0.3333, 0.3333]], dtype=torch.float32)
    assert torch.allclose(result, expected, atol=1e-4)


def test_sparsemax_forward_dim_handling():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    result = SparsemaxFunction.apply(x, 0)
    assert result.shape == x.shape
    assert torch.all(result >= 0)


def test_sparsemax_idempotence():
    x = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], dtype=torch.float32)
    y1 = SparsemaxFunction.apply(x)
    y2 = SparsemaxFunction.apply(y1)
    assert torch.allclose(y1, y2, atol=1e-5), "Sparsemax should be idempotent (applying twice yields same result)"


@pytest.mark.parametrize(
    "device", [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
)
def test_sparsemax_device_consistency(device):
    x = torch.randn(5, 3, dtype=torch.float32, device=device, requires_grad=True)
    y = SparsemaxFunction.apply(x)
    assert y.device == x.device
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.device == x.device


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_sparsemax_dtype_consistency(dtype):
    x = torch.randn(5, 3, dtype=dtype, requires_grad=True)
    y = SparsemaxFunction.apply(x)
    assert y.dtype == x.dtype


@pytest.mark.parametrize("shape", [(0, 3), (1, 3), (3, 0)])
def test_sparsemax_empty_and_singleton(shape):
    x = torch.empty(*shape, dtype=torch.float32, requires_grad=True)
    y = SparsemaxFunction.apply(x)
    assert y.shape == x.shape


@pytest.mark.parametrize(
    "x",
    [
        torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32),
        torch.tensor([[-1.0, -2.0, -3.0], [0.5, 0.5, 0.5]], dtype=torch.float32),
    ],
)
def test_sparsemax_monotonicity(x):
    # Monotonicity: increasing input should not decrease output
    x2 = x + 1.0
    y1 = SparsemaxFunction.apply(x)
    y2 = SparsemaxFunction.apply(x2)
    assert torch.all(y2 >= y1 - 1e-5), "Sparsemax output should be monotonic with respect to input"


@pytest.mark.parametrize(
    "x",
    [
        torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),  # simple ascending input
        torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float32),  # simple descending input
        torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32),  # uniform input
        torch.tensor([[1.0, -1.0, 0.0]], dtype=torch.float32),  # mixed positive and negative
        torch.tensor([[10.0, 10.0, 10.0]], dtype=torch.float32),  # large uniform values
        torch.tensor([[10.0, 10.0, 10.0]], dtype=torch.float32),  # large uniform values
        torch.zeros((1, 3), dtype=torch.float32),  # zeros input
    ],
)
def test_sparsemax_backward_parametrized(x):
    x = x.clone().detach().requires_grad_(True)
    result = SparsemaxFunction.apply(x)
    result.sum().backward()

    assert x.grad is not None, "Gradient should not be None"
    assert x.grad.shape == x.shape, "Gradient should have same shape as input"

    # Verify gradients sum to zero per sparsemax constraints
    grad_sum = x.grad.sum(-1)

    assert torch.allclose(grad_sum, torch.zeros_like(grad_sum), atol=1e-5), (
        "Gradients should sum to zero along the projection dimension"
    )
