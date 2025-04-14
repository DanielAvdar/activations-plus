# Implementation of the SReLU (S-shaped ReLU) function
import torch


class SReLU(torch.nn.Module):
    def __init__(self, lower_threshold=-1.0, upper_threshold=1.0):
        super(SReLU, self).__init__()
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def forward(self, x):
        return torch.where(
            x < self.lower_threshold,
            self.lower_threshold,
            torch.where(x > self.upper_threshold, self.upper_threshold, x),
        )
