import torch


class HardSwish(torch.nn.Module):
    """
    HardSwish activation function.

    An efficient approximation of the Swish activation function,
    often used in mobile and embedded applications.
    """

    def forward(self, x):
        return x * torch.clamp((x + 3) / 6, min=0, max=1)
