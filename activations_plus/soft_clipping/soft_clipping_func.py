import torch


class SoftClipping(torch.nn.Module):
    """
    Soft Clipping activation function.

    This activation function smoothly limits the range of activations, preventing extreme values
    without hard truncation. It is particularly useful for stabilizing neural network training.

    Attributes:
        min_val (float): The minimum value of the activation range.
        max_val (float): The maximum value of the activation range.

    Methods:
        forward(x): Computes the Soft Clipping activation for the input tensor `x`.
    """

    def __init__(self, min_val=-1.0, max_val=1.0):
        super(SoftClipping, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return self.min_val + (self.max_val - self.min_val) * torch.sigmoid(x)
