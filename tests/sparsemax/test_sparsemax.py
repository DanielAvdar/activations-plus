import pytest
import torch

from activations_plus import Sparsemax


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



@pytest.mark.parametrize("dim", [-1, 0, 1])
@pytest.mark.parametrize("input_shape", [(5, 3), (2, 4, 6), (2, 2, 3, 4), (10, 20, 30)])
def test_sparsemax(input_shape, dim):
    input_ = torch.randn(input_shape)
    sparsemax_op = Sparsemax(dim=dim)
    output = sparsemax_op(input_)

    # Check shape
    assert output.shape == input_.shape

    # Check normalization along the specified dimension
    sum_along_dim = output.sum(dim=dim)
    assert torch.allclose(sum_along_dim, torch.ones_like(sum_along_dim)), (
        f"Output does not sum to 1 along dimension {dim}"
    )

    # Check sparsity (some elements should be exactly zero)
    assert (output == 0).any()