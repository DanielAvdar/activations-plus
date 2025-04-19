"""Implements the Sparsemax function for PyTorch."""

from typing import Any

import torch

from activations_plus.sparsemax.utils import flatten_all_but_nth_dim, unflatten_all_but_nth_dim


class SparsemaxFunction(torch.autograd.Function):
    """Sparsemax autograd function for forward and backward passes."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Perform the forward pass for SparsemaxFunction with arbitrary dim support."""
        input_dim = x.dim()
        if input_dim <= dim or dim < -input_dim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [-{input_dim}, {input_dim - 1}], but got {dim})"
            )

        # Save operating dimension to context
        ctx.needs_reshaping = input_dim > 2 and dim != (input_dim - 1) and dim != -1
        ctx.dim = dim

        if ctx.needs_reshaping:
            ctx, x = flatten_all_but_nth_dim(ctx, x)
            # After flatten, the dim of interest is now 1
            reduce_dim = 1
        else:
            reduce_dim = dim

        # Translate by max for numerical stability
        x = x - x.max(dim=reduce_dim, keepdim=True).values.expand_as(x)

        zs = x.sort(dim=reduce_dim, descending=True).values
        d = x.size(reduce_dim)
        range_th = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
        shape = [1] * x.dim()
        shape[reduce_dim] = d
        range_th = range_th.view(*shape).expand_as(x)

        # Determine sparsity of projection
        bound = 1 + range_th * zs
        cumsum_zs = zs.cumsum(dim=reduce_dim)
        is_gt = bound.gt(cumsum_zs).type(x.dtype)
        k = (is_gt * range_th).max(dim=reduce_dim, keepdim=True).values

        # Compute threshold
        zs_sparse = is_gt * zs
        taus = (zs_sparse.sum(dim=reduce_dim, keepdim=True) - 1) / k
        taus = taus.expand_as(x)

        output = torch.max(torch.zeros_like(x), x - taus)

        # Save context
        ctx.save_for_backward(output)

        # Reshape back to original shape
        if ctx.needs_reshaping:
            ctx, output = unflatten_all_but_nth_dim(ctx, output)

        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore
        """Compute the backward pass for SparsemaxFunction."""
        output, *_ = ctx.saved_tensors

        if ctx.needs_reshaping:
            ctx, grad_output = flatten_all_but_nth_dim(ctx, grad_output)
            reduce_dim = 1
        else:
            reduce_dim = ctx.dim

        nonzeros = torch.ne(output, 0)
        num_nonzeros = nonzeros.sum(dim=reduce_dim, keepdim=True)
        sum_all = (grad_output * nonzeros).sum(dim=reduce_dim, keepdim=True) / num_nonzeros
        grad_input = nonzeros * (grad_output - sum_all.expand_as(grad_output))

        if ctx.needs_reshaping:
            ctx, grad_input = unflatten_all_but_nth_dim(ctx, grad_input)

        return grad_input, None
