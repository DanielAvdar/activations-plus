import torch


class ELiSH(torch.nn.Module):
    """
    ELiSH (Exponential Linear Sigmoid Squash) activation function.

    Combines properties of exponential and sigmoid functions,
     aiming to retain small negative values while maintaining smoothness.
    """

    def forward(self, x):
        return torch.where(x > 0, x / (1 + torch.exp(-x)), torch.exp(x) - 1)
