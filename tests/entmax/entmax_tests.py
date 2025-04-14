import pytest
import torch

from activations_plus.entmax.entmax import Entmax


def test_entmax15_forward():
    entmax = Entmax(dim=-1)
    input_tensor = torch.tensor([[1.0, 2.0, 0.5], [3.0, 0.0, 0.1]])
    output = entmax(input_tensor)

    assert output.shape == input_tensor.shape
    assert (output >= 0).all()
    assert torch.allclose(output.sum(-1), torch.ones_like(output.sum(-1)))


def test_entmax15_dim_validation():
    entmax = Entmax(dim=5)
    input_tensor = torch.tensor([[1.0, 2.0, 0.5]])
    with pytest.raises(IndexError, match="Dimension out of range"):
        entmax(input_tensor)


def test_entmax15_extra_repr():
    entmax = Entmax(dim=0)
    assert entmax.extra_repr() == "dim=0"
