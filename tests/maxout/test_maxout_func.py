import torch

from activations_plus.maxout.maxout_func import Maxout


def test_maxout():
    activation = Maxout(num_pieces=2)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = activation(x)
    expected = torch.max(x.view(2, 1, 2), dim=-1)[0]
    assert torch.allclose(y, expected)
