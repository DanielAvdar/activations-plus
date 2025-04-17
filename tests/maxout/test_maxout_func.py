import pytest
import torch

from activations_plus.maxout.maxout_func import Maxout


def test_maxout():
    activation = Maxout(num_pieces=2)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = activation(x)
    expected = torch.max(x.view(2, 1, 2), dim=-1)[0]
    assert torch.allclose(y, expected)


def test_maxout_math():
    activation = Maxout(num_pieces=2)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = activation(x)
    expected = torch.max(x.view(2, 1, 2), dim=-1)[0]
    assert torch.allclose(y, expected), "Maxout does not match the mathematical definition."


@pytest.mark.parametrize(
    "x, expected",
    [
        (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.max(torch.tensor([[1.0, 2.0], [3.0, 4.0]]).view(2, 1, 2), dim=-1)[0],
        ),
        (
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            torch.max(torch.tensor([[0.0, 0.0], [0.0, 0.0]]).view(2, 1, 2), dim=-1)[0],
        ),
        (
            torch.tensor([[-1.0, -2.0], [-3.0, -4.0]]),
            torch.max(torch.tensor([[-1.0, -2.0], [-3.0, -4.0]]).view(2, 1, 2), dim=-1)[0],
        ),
    ],
)
def test_maxout_math_param(x, expected):
    activation = Maxout(num_pieces=2)
    y = activation(x)
    assert torch.allclose(y, expected), "Maxout does not match the mathematical definition."
