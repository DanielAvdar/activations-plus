import torch.nn as nn

from .sparsemax_func import SparsemaxFunction


class Sparsemax(nn.Module):
    """
    Sparsemax class implements a transformation function from the paper
    "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label
    Classification" (https://arxiv.org/pdf/1602.02068.pdf). This function is
    used as an activation function that is similar to softmax but can produce
    sparse output, where some of the entries are exactly zero.

    This class is designed to handle the computation over a specified
    dimension, and it can be used as a module in neural network architectures.

    :ivar dim: The dimension over which to apply the Sparsemax operation. If not provided,
        the default is -1.
    :type dim: int
    """

    __constants__ = ["dim"]

    def __init__(self, dim=-1):
        """
        Sparsemax class as seen in https://arxiv.org/pdf/1602.02068.pdf
        Parameters
        ----------
        dim: The dimension we want to cast the operation over. Default -1
        """
        super(Sparsemax, self).__init__()
        self.dim = dim

    #
    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     if not hasattr(self, "dim"):
    #         self.dim = None

    def forward(self, x):
        """
        Applies the sparsemax function along the specified dimension.

        Sparsemax is a neural network activation function that maps input logits
        to probabilities, similar to softmax. Unlike softmax, it can lead to sparse
        probability distributions where some probabilities are exactly zero.

        :param x: The input tensor to which the sparsemax function will be applied.
        :return: The tensor after applying the sparsemax operation along the specified
            dimension.
        """
        return SparsemaxFunction.apply(x, self.dim)

    def extra_repr(self):
        return f"dim={self.dim}"
