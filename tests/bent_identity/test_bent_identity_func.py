import pytest
import torch

from activations_plus.bent_identity.bent_identity_func import BentIdentity


def test_bent_identity():
    activation = BentIdentity()
    x = torch.tensor([-3.0, 0.0, 3.0])
    y = activation(x)
    expected = (torch.sqrt(x**2 + 1) - 1) / 2 + x
    assert torch.allclose(y, expected)


def test_bent_identity_math():
    activation = BentIdentity()
    x = torch.tensor([-3.0, 0.0, 3.0])
    y = activation(x)
    expected = (torch.sqrt(x**2 + 1) - 1) / 2 + x
    assert torch.allclose(y, expected), "Bent Identity does not match the mathematical definition."


@pytest.mark.parametrize(
    "x, expected",
    [
        (torch.tensor([-3.0]), (torch.sqrt(torch.tensor([-3.0]) ** 2 + 1) - 1) / 2 + torch.tensor([-3.0])),
        (torch.tensor([0.0]), (torch.sqrt(torch.tensor([0.0]) ** 2 + 1) - 1) / 2 + torch.tensor([0.0])),
        (torch.tensor([3.0]), (torch.sqrt(torch.tensor([3.0]) ** 2 + 1) - 1) / 2 + torch.tensor([3.0])),
    ],
)
def test_bent_identity_math_param(x, expected):
    activation = BentIdentity()
    y = activation(x)
    assert torch.allclose(y, expected), "Bent Identity does not match the mathematical definition."
