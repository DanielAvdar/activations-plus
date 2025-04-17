import pytest
import torch

from activations_plus.elish.elish_func import ELiSH


def test_elish_positive():
    activation = ELiSH()
    x = torch.tensor([1.0, 2.0, 3.0])
    y = activation(x)
    assert torch.allclose(y, x / (1 + torch.exp(-x)))


def test_elish_negative():
    activation = ELiSH()
    x = torch.tensor([-1.0, -2.0, -3.0])
    y = activation(x)
    assert torch.allclose(y, torch.exp(x) - 1)


def test_elish_math():
    activation = ELiSH()
    x = torch.tensor([-3.0, 0.0, 3.0])
    y = activation(x)
    expected = torch.where(x > 0, x / (1 + torch.exp(-x)), torch.exp(x) - 1)
    assert torch.allclose(y, expected), "ELiSH does not match the mathematical definition."


@pytest.mark.parametrize(
    "x, expected",
    [
        (torch.tensor([-3.0]), torch.exp(torch.tensor([-3.0])) - 1),
        (torch.tensor([0.0]), torch.tensor([0.0]) / (1 + torch.exp(torch.tensor([0.0])))),
        (torch.tensor([3.0]), torch.tensor([3.0]) / (1 + torch.exp(-torch.tensor([3.0])))),
    ],
)
def test_elish_math_param(x, expected):
    activation = ELiSH()
    y = activation(x)
    assert torch.allclose(y, expected), "ELiSH does not match the mathematical definition."
