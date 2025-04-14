import torch


class BentIdentity(torch.nn.Module):
    """
    Bent Identity activation function.

    This activation function provides a smooth approximation of the identity function.
    It introduces non-linearity while preserving the identity mapping for large inputs.

    Attributes:
        None

    Methods:
        forward(x): Computes the Bent Identity activation for the input tensor `x`.
    """

    def forward(self, x):
        return (torch.sqrt(x**2 + 1) - 1) / 2 + x
