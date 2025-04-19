"""Sparsemax activation package initialization."""

from .sparsemax_func_v2 import SparsemaxFunction
from .sparsemax_v2 import Sparsemax
from .utils import flatten_all_but_nth_dim, unflatten_all_but_nth_dim

__all__ = ["SparsemaxFunction", "Sparsemax", "flatten_all_but_nth_dim", "unflatten_all_but_nth_dim"]
