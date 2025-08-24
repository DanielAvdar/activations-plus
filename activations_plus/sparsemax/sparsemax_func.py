"""Implements the Sparsemax function for PyTorch."""

from typing import Any

import torch

from activations_plus.sparsemax.utils import flatten_all_but_nth_dim, unflatten_all_but_nth_dim


class SparsemaxFunction(torch.autograd.Function):
    """Sparsemax autograd function for forward and backward passes."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Perform the forward pass for SparsemaxFunction with arbitrary dim support.

        Parameters
        ----------
        ctx : Any
            Context object for autograd.
        x : torch.Tensor
            Input tensor.
        dim : int, optional
            Dimension along which to apply Sparsemax. Default is -1.

        Returns
        -------
        torch.Tensor
            The output tensor after applying Sparsemax.

        Raises
        ------
        IndexError
            If the specified dimension is out of range for the input tensor.

        """
        # Handle empty input: return empty tensor of same shape/type
        if x.numel() == 0:
            return x.clone()

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

        output, ctx = SparsemaxFunction._threshold_and_support(ctx, x, reduce_dim)

        # Save context
        ctx.save_for_backward(output)

        # Reshape back to original shape
        if ctx.needs_reshaping:
            ctx, output = unflatten_all_but_nth_dim(ctx, output)

        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore
        """Compute the backward pass for SparsemaxFunction.

        Parameters
        ----------
        ctx : Any
            Context object for autograd.
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output.

        Returns
        -------
        tuple[torch.Tensor, None]
            The gradient with respect to the input and None for dim.

        """
        # Handle empty input: return zero-like grad_input and None
        if grad_output.numel() == 0:
            return grad_output.clone(), None

        output, *_ = ctx.saved_tensors

        if ctx.needs_reshaping:
            ctx, grad_output = flatten_all_but_nth_dim(ctx, grad_output)
            reduce_dim = 1
        else:
            reduce_dim = ctx.dim

        grad_input = SparsemaxFunction._compute_gradient(ctx, grad_output, output, reduce_dim)

        if ctx.needs_reshaping:
            ctx, grad_input = unflatten_all_but_nth_dim(ctx, grad_input)

        return grad_input, None

    @staticmethod
    def _threshold_and_support(ctx: Any, x: torch.Tensor, reduce_dim: int) -> tuple[torch.Tensor, Any]:
        """Compute the threshold and support for the input tensor.

        Parameters
        ----------
        ctx : Any
            Context object for autograd.
        x : torch.Tensor
            Input tensor.
        reduce_dim : int
            Dimension along which to compute threshold/support.

        Returns
        -------
        tuple[torch.Tensor, Any]
            The output tensor after applying Sparsemax and the updated context.

        """
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

        return output, ctx

    @staticmethod
    def _compute_gradient(ctx: Any, grad_output: torch.Tensor, output: torch.Tensor, reduce_dim: int) -> torch.Tensor:
        """Compute the gradient for the backward pass.

        Parameters
        ----------
        ctx : Any
            Context object for autograd.
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output.
        output : torch.Tensor
            Output tensor from the forward pass.
        reduce_dim : int
            Dimension along which to compute the gradient.

        Returns
        -------
        torch.Tensor
            The gradient with respect to the input.

        """
        nonzeros = torch.ne(output, 0)
        num_nonzeros = nonzeros.sum(dim=reduce_dim, keepdim=True)
        sum_all = (grad_output * nonzeros).sum(dim=reduce_dim, keepdim=True) / num_nonzeros
        grad_input = nonzeros * (grad_output - sum_all.expand_as(grad_output))

        return grad_input
