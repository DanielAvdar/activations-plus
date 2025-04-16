import pytest
import torch

from activations_plus.soft_clipping.soft_clipping_func import SoftClipping


def test_soft_clipping():
    activation = SoftClipping(min_val=-1.0, max_val=1.0)
    x = torch.tensor([-3.0, 0.0, 3.0])
    y = activation(x)
    expected = -1.0 + (1.0 - (-1.0)) * torch.sigmoid(x)
    assert torch.allclose(y, expected)


def test_soft_clipping_math():
    activation = SoftClipping(min_val=-1.0, max_val=1.0)
    x = torch.tensor([-3.0, 0.0, 3.0])
    y = activation(x)
    expected = -1.0 + (1.0 - (-1.0)) * torch.sigmoid(x)
    assert torch.allclose(y, expected), "Soft Clipping does not match the mathematical definition."


@pytest.mark.parametrize(
    "x, expected",
    [
        (torch.tensor([-3.0]), -1.0 + (1.0 - (-1.0)) * torch.sigmoid(torch.tensor([-3.0]))),
        (torch.tensor([0.0]), -1.0 + (1.0 - (-1.0)) * torch.sigmoid(torch.tensor([0.0]))),
        (torch.tensor([3.0]), -1.0 + (1.0 - (-1.0)) * torch.sigmoid(torch.tensor([3.0]))),
    ],
)
def test_soft_clipping_math_param(x, expected):
    activation = SoftClipping(min_val=-1.0, max_val=1.0)
    y = activation(x)
    assert torch.allclose(y, expected), "Soft Clipping does not match the mathematical definition."
