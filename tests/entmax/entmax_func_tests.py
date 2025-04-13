import pytest
import torch

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
