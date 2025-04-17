import torch


class Maxout(torch.nn.Module):
    """Maxout activation function.

    Selects the maximum across multiple linear functions,
    allowing the network to learn piecewise linear convex functions.
    """

    def __init__(self, num_pieces: int) -> None:
        """Represents a Maxout activation module used in neural networks. Maxout activation
        splits the input into multiple pieces and selects the maximum value from each
        set of pieces.

        This module is useful for creating complex, non-linear decision boundaries in
        machine learning models. Each instance of this class initializes the number of
        pieces into which the input is split, used to perform the Maxout operation.

        :param num_pieces: Number of pieces into which the input is divided for the
            Maxout operation.
        """
        super(Maxout, self).__init__()
        self.num_pieces = num_pieces

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reshapes the input tensor to split its last dimension into multiple parts and then
        computes the maximum along the newly added dimension.

        The function modifies the shape of the input tensor such that the last dimension is
        divided into `num_pieces`. It then computes and returns the maximum values along the
        last axis of the reshaped tensor.

        :param x: A tensor of arbitrary shape where the last dimension must be divisible by
            `self.num_pieces`.
        :return: A tensor containing the maximum values along the split dimension of the
            reshaped input tensor. The resulting shape will match all but the last dimension
            of the input tensor.
        """
        shape = x.shape[:-1] + (x.shape[-1] // self.num_pieces, self.num_pieces)
        x = x.view(*shape)
        return x.max(-1)[0]
