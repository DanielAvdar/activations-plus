import pytest
import torch

from activations_plus.sparsemax.sparsemax import Sparsemax


def test_sparsemax_forward():
    sparsemax = Sparsemax(dim=-1)
    input_tensor = torch.tensor([[1.0, 2.0, 0.5], [3.0, 0.0, 0.1]])
    output = sparsemax(input_tensor)

    assert output.shape == input_tensor.shape
    assert (output >= 0).all()


def test_sparsemax_dim_validation():
    sparsemax = Sparsemax(dim=5)
    input_tensor = torch.tensor([[1.0, 2.0, 0.5]])
    with pytest.raises(IndexError, match="Dimension out of range"):
        sparsemax(input_tensor)
