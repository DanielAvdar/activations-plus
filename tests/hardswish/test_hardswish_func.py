import torch

from activations_plus.hardswish.hardswish_func import HardSwish


def test_hardswish():
    activation = HardSwish()
    x = torch.tensor([-3.0, 0.0, 3.0])
    y = activation(x)
    expected = x * torch.clamp((x + 3) / 6, min=0, max=1)
    assert torch.allclose(y, expected)
