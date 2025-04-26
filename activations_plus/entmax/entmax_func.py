"""Implements the Entmax15 function for PyTorch."""

from typing import Any

import torch


class Entmax15Function(torch.autograd.Function):
    """Implement exact Entmax with alpha=1.5 (B. Peters, V. Niculae, A. Martins).

    See :cite:`https://arxiv.org/abs/1905.05702` for detailed description. Source: https://github.com/deep-spin/entmax.
    """

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Perform the forward pass for Entmax15Function.

        Parameters
        ----------
        ctx : Any
            Context object for autograd.
        input_ : torch.Tensor
            Input tensor.
        dim : int, optional
            Dimension along which to apply Entmax15. Default is -1.

        Returns
        -------
        torch.Tensor
            The output tensor after applying Entmax15.

        """
        ctx.dim = dim

        # Handle empty input early (like softmax)
        if input_.numel() == 0:
            return input_.clone()

        max_val, _ = input_.max(dim=dim, keepdim=True)
        input_ = input_ - max_val  # same numerical stability trick as for softmax
        input_ = input_ / 2  # divide by 2 to solve actual Entmax

        tau_star, _ = Entmax15Function._threshold_and_support(input_, dim)
        output = torch.clamp(input_ - tau_star, min=0) ** 2
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore
        """Perform the backward pass for Entmax15Function.

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
        # Handle empty grad_output (empty input case)
        if grad_output.numel() == 0:
            return grad_output.clone(), None
        (y,) = ctx.saved_tensors
        gppr = y.sqrt()  # = 1 / g'' (y)
        dx = grad_output * gppr
        q = dx.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dx -= q * gppr
        return dx, None

    @staticmethod
    def _make_ix_like(input_: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Create an index tensor like the input tensor.

        Parameters
        ----------
        input_ : torch.Tensor
            Input tensor.
        dim : int, optional
            Dimension for the index tensor. Default is 0.

        Returns
        -------
        torch.Tensor
            Index tensor with the same device and dtype as input_.

        """
        d = input_.size(dim)
        rho = torch.arange(1, d + 1, device=input_.device, dtype=input_.dtype)
        view = [1] * input_.dim()
        view[0] = -1
        return rho.view(view).transpose(0, dim)

    @staticmethod
    def _threshold_and_support(input_: torch.Tensor, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the threshold and support for the input tensor.

        Parameters
        ----------
        input_ : torch.Tensor
            Input tensor.
        dim : int, optional
            Dimension along which to compute threshold/support. Default is -1.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The threshold and support tensors.

        """
        xsrt, _ = torch.sort(input_, descending=True, dim=dim)

        rho = Entmax15Function._make_ix_like(input_, dim)
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
