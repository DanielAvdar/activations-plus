import torch

from activations_plus.sparsemax.utils import flatten_all_but_nth_dim, unflatten_all_but_nth_dim


class Entmax15Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int = -1):
        input_dim = x.dim()
        if input_dim <= dim or dim < -input_dim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [-{input_dim}, {input_dim - 1}], but got {dim})"
            )

        ctx.needs_reshaping = input_dim > 2
        ctx.dim = dim

        if ctx.needs_reshaping:
            ctx, x = flatten_all_but_nth_dim(ctx, x)

        max_val, _ = x.max(dim=-1, keepdim=True)
        x = x - max_val  # Numerical stability

        tau, supp_size = Entmax15Function._threshold_and_support(x)
        output = torch.clamp((x - tau).pow(2), min=0.0)

        ctx.save_for_backward(output.sqrt(), supp_size)

        output /= output.sum(dim=-1, keepdim=True)  # Normalization

        if ctx.needs_reshaping:
            ctx, output = unflatten_all_but_nth_dim(ctx, output)

        return output

    @staticmethod
    def _threshold_and_support(x):
        x_sorted, _ = torch.sort(x, descending=True, dim=-1)
        rho = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype)
        mean_cumsum = (x_sorted.cumsum(dim=-1) - 1) / rho
        mean_sq = (x_sorted**2).cumsum(dim=-1)
        mean_sq = mean_sq / rho
        ss = rho * (mean_sq - mean_cumsum**2)
        delta = (1 - ss) / rho
        support = (delta > 0).type(x.dtype)
        supp_size = support.sum(dim=-1, keepdim=True).long()
        tau = mean_cumsum.gather(dim=-1, index=supp_size - 1)
        return tau, supp_size

    @staticmethod
    def backward(ctx, grad_output):
        y_sqrt, supp_size = ctx.saved_tensors
        grad_input = grad_output.clone()

        nonzeros = (y_sqrt > 0).type(grad_output.dtype)
        sum_grad = (grad_input * y_sqrt).sum(dim=-1, keepdim=True)
        sum_grad /= y_sqrt.sum(dim=-1, keepdim=True)

        grad_input = 2 * y_sqrt * (grad_input - sum_grad)
        grad_input = nonzeros * grad_input

        if ctx.needs_reshaping:
            ctx, grad_input = unflatten_all_but_nth_dim(ctx, grad_input)

        return grad_input, None
