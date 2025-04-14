# Implementation of the SReLU (S-shaped ReLU), ELiSH (Exponential Linear Sigmoid Squash), and Soft Clipping functions
import torch


class SReLU(torch.nn.Module):
    """
    SReLU (S-shaped Rectified Linear Unit) activation function.

    Args:
        lower_threshold (float): The lower threshold value. Defaults to -1.0.
        upper_threshold (float): The upper threshold value. Defaults to 1.0.

    Raises:
        ValueError: If lower_threshold is greater than upper_threshold.
    """

    def __init__(self, lower_threshold=-1.0, upper_threshold=1.0):
        super(SReLU, self).__init__()
        if lower_threshold > upper_threshold:
            raise ValueError("lower_threshold must be less than or equal to upper_threshold")
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def forward(self, x):
        """
        Forward pass of the SReLU activation function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying SReLU.
        """
        return torch.where(
            x < self.lower_threshold,
            self.lower_threshold,
            torch.where(x > self.upper_threshold, self.upper_threshold, x),
        )

    def extra_repr(self):
        """Extra representation for printing the module."""
        return f"lower_threshold={self.lower_threshold}, upper_threshold={self.upper_threshold}"
