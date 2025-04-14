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
