import torch


class SoftClipping(torch.nn.Module):
    """
    Soft Clipping activation function.

    Limits the range of activations smoothly,
    preventing extreme values without hard truncation.
    """

    def __init__(self, min_val=-1.0, max_val=1.0):
        super(SoftClipping, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return self.min_val + (self.max_val - self.min_val) * torch.sigmoid(x)
