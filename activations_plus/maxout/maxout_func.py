import torch


class Maxout(torch.nn.Module):
    """
    Maxout activation function.

    Selects the maximum across multiple linear functions,
    allowing the network to learn piecewise linear convex functions.
    """

    def __init__(self, num_pieces: int) -> None:
        super(Maxout, self).__init__()
        self.num_pieces = num_pieces

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1] + (x.shape[-1] // self.num_pieces, self.num_pieces)
        x = x.view(*shape)
        return x.max(-1)[0]
