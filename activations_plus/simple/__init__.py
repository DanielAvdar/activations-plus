"""Simple activation functions for PyTorch.

This package provides basic activation functions for demonstration and testing.
"""


def identity(x):
    """Return the input unchanged.

    Parameters
    ----------
    x : Any
        Input value.

    Returns
    -------
    Any
        The input value unchanged.

    """
    return x


def binary_step(x):
    """Return 1 if input is non-negative, else 0.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    int
        1 if x >= 0, else 0.

    """
    return 1 if x >= 0 else 0
