import torch
import torch.nn.functional as functional
from torch import nn
from torch.autograd import Function

"""
Other possible implementations:
https://github.com/KrisKorrel/sparsemax-pytorch/blob/master/sparsemax.py
https://github.com/msobroza/SparsemaxPytorch/blob/master/mnist/sparsemax.py
https://github.com/vene/sparse-structured-attention/blob/master/pytorch/torchsparseattn/sparsemax.py
"""


# credits to Yandex https://github.com/Qwicen/node/blob/master/lib/nn_utils.py
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


class Entmax15Function(Function):
    """
    An implementation of exact Entmax with alpha=1.5 (B. Peters, V. Niculae, A. Martins). See
    :cite:`https://arxiv.org/abs/1905.05702 for detailed description.
    Source: https://github.com/deep-spin/entmax
    """

    @staticmethod
    def forward(ctx: torch.Tensor, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        ctx.dim = dim

        max_val, _ = input_.max(dim=dim, keepdim=True)
        input_ = input_ - max_val  # same numerical stability trick as for softmax
        input_ = input_ / 2  # divide by 2 to solve actual Entmax

        tau_star, _ = Entmax15Function._threshold_and_support(input_, dim)
        output = torch.clamp(input_ - tau_star, min=0) ** 2
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx: torch.Tensor, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (y,) = ctx.saved_tensors
        gppr = y.sqrt()  # = 1 / g'' (y)
        dx = grad_output * gppr
        q = dx.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dx -= q * gppr
        return dx, None

    @staticmethod
    def _threshold_and_support(input_: torch.Tensor, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
        xsrt, _ = torch.sort(input_, descending=True, dim=dim)

        rho = _make_ix_like(input_, dim)
        mean = xsrt.cumsum(dim) / rho
        mean_sq = (xsrt**2).cumsum(dim) / rho
        ss = rho * (mean_sq - mean**2)
        delta = (1 - ss) / rho

        # NOTE this is not exactly the same as in reference algo
        # Fortunately it seems the clamped values never wrongly
        # get selected by tau <= sorted_z. Prove this!
        delta_nz = torch.clamp(delta, 0)
        tau = mean - torch.sqrt(delta_nz)

        support_size = (tau <= xsrt).sum(dim).unsqueeze(dim)
        tau_star = tau.gather(dim, support_size - 1)
        return tau_star, support_size


class Entmoid15(Function):
    """A highly optimized equivalent of lambda x: Entmax15([x, 0])"""

    @staticmethod
    def forward(ctx: torch.Tensor, input_: torch.Tensor) -> torch.Tensor:
        output = Entmoid15._forward(input_)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def _forward(input_: torch.Tensor) -> torch.Tensor:
        input_, is_pos = abs(input_), input_ >= 0
        tau = (input_ + torch.sqrt(functional.relu(8 - input_**2))) / 2
        tau.masked_fill_(tau <= input_, 2.0)
        y_neg = 0.25 * functional.relu(tau - input_, inplace=True) ** 2
        return torch.where(is_pos, 1 - y_neg, y_neg)

    @staticmethod
    def backward(ctx: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
        return Entmoid15._backward(ctx.saved_tensors[0], grad_output)

    @staticmethod
    def _backward(output: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
        gppr0, gppr1 = output.sqrt(), (1 - output).sqrt()
        grad_input = grad_output * gppr0
        q = grad_input / (gppr0 + gppr1)
        grad_input -= q * gppr0
        return grad_input


entmax15 = Entmax15Function.apply
entmoid15 = Entmoid15.apply


class Entmax15(nn.Module):
    def __init__(self, dim: int = -1):
        self.dim = dim
        super(Entmax15, self).__init__()

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return entmax15(input_, self.dim)
