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

    def __init__(self, x_min=-1.0, x_max=1.0):
        super(SoftClipping, self).__init__()
        self.min_val = x_min
        self.max_val = x_max

    def forward(self, x):
        return self.min_val + (self.max_val - self.min_val) * torch.sigmoid(x)
