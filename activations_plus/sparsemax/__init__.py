"""Sparsemax activation package initialization."""

from .sparsemax import Sparsemax
from .sparsemax_func import SparsemaxFunction
from .utils import flatten_all_but_nth_dim, unflatten_all_but_nth_dim

__all__ = ["SparsemaxFunction", "Sparsemax", "flatten_all_but_nth_dim", "unflatten_all_but_nth_dim"]
