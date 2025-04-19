"""Implements the Sparsemax activation function for PyTorch."""

import torch
from torch import nn
from torch.autograd import Function

def _make_ix_like(input_: torch.Tensor, dim: int = 0) -> torch.Tensor:
    d = input_.size(dim)
    rho = torch.arange(1, d + 1, device=input_.device, dtype=input_.dtype)
    view = [1] * input_.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


class SparsemaxFunction(Function):
    """
    An implementation of sparsemax (Martins & Astudillo, 2016). See
    :cite:`DBLP:journals/corr/MartinsA16` for detailed description.
    By Ben Peters and Vlad Niculae
    """

    @staticmethod
    def forward(ctx: torch.Tensor, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """sparsemax: normalizing sparse transform (a la softmax)

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
        input_ : torch.Tensor
            any shape
        dim : int
            dimension along which to apply sparsemax

        Returns
        -------
        output : torch.Tensor
            same shape as input_

        """
        ctx.dim = dim
        max_val, _ = input_.max(dim=dim, keepdim=True)
        input_ -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = SparsemaxFunction._threshold_and_support(input_, dim=dim)
        output = torch.clamp(input_ - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx: torch.Tensor, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

    @staticmethod
    def _threshold_and_support(input_: torch.Tensor, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
        """Sparsemax building block: compute the threshold

        Parameters
        ----------
        input_: torch.Tensor
            any dimension
        dim : int
            dimension along which to apply the sparsemax

        Returns
        -------
        tau : torch.Tensor
            the threshold value
        support_size : torch.Tensor

        """

        input_srt, _ = torch.sort(input_, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = _make_ix_like(input_, dim)
        support = rhos * input_srt > input_cumsum

        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input_.dtype)
        return tau, support_size


sparsemax = SparsemaxFunction.apply


class Sparsemax(nn.Module):
    def __init__(self, dim: int = -1):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return sparsemax(input_, self.dim)