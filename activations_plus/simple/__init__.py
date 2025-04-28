"""Simple activation functions and their variants for neural networks."""

from .elu_variants import isrlu
from .relu_variants import dual_line
from .sigmoid_tanh_variants import aria2, isru, tanh_exp
from .specialized_variants import (
    erf_act,
    hat,
    resp,
)
from .tanh_variants import penalized_tanh

__all__ = [
    # ReLU variants
    "dual_line",
    # Sigmoid/Tanh variants
    "tanh_exp",
    "isru",
    # Polynomial/Power variants
    # Log/Exp/Softplus variants
    # ELU variants
    "isrlu",
    # GELU/Swish variants
    # Tanh variants
    "penalized_tanh",
    # Specialized variants
    "erf_act",
    "hat",
    "resp",
    "aria2",
]
