import pytest
import torch
from torch.autograd import gradcheck

from activations_plus.sparsemax import Sparsemax


@pytest.mark.parametrize("dimension", [-4, -3, -2, -1, 0, 1, 2, 3])
# @pytest.mark.skip(reason="Skipping grad-check temporarily")
def test_sparsemax_grad(dimension):
    sparsemax = Sparsemax(dimension)
    inputs = torch.randn(6, 3, 5, 4, dtype=torch.double, requires_grad=True)
    assert gradcheck(sparsemax, inputs, eps=1e-6, atol=1e-4)


def test_sparsemax_invalid_dimension():
    sparsemax = Sparsemax(-7)
    inputs = torch.randn(6, 3, 5, 4, dtype=torch.double, requires_grad=True)
    with pytest.raises(IndexError):
        gradcheck(sparsemax, inputs, eps=1e-6, atol=1e-4)
