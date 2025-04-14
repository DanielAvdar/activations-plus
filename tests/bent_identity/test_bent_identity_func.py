import torch

from activations_plus.bent_identity.bent_identity_func import BentIdentity


def test_bent_identity():
    activation = BentIdentity()
    x = torch.tensor([-3.0, 0.0, 3.0])
    y = activation(x)
    expected = (torch.sqrt(x**2 + 1) - 1) / 2 + x
    assert torch.allclose(y, expected)
