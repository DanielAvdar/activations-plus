"""Simple activation functions and their variants for neural networks."""

from .elu_variants import isrlu, pelu
from .gelu_swish_variants import swish
from .relu_variants import dual_line
from .sigmoid_tanh_variants import aria2, isru, tanh_exp
from .specialized_variants import (
    erf_act,
    hat,
    pserf,
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
    "pelu",
    # GELU/Swish variants
    "swish",
    # Tanh variants
    "penalized_tanh",
    # Specialized variants
    "erf_act",
    "hat",
    "resp",
    "aria2",
    "pserf",
]
