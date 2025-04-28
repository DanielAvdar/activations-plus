"""Simple activation functions and their variants for neural networks."""

from .elu_variants import abslu, celu, elu, isrlu, selu
from .gelu_swish_variants import gelu, hard_sigmoid, hard_swish, mish, phish, silu
from .log_exp_softplus_variants import logish, loglog, loglogish, soft_exponential, softplus_linear_unit
from .polynomial_power_variants import (
    inverse_polynomial_linear_unit,
    polynomial_linear_unit,
    power_function_linear_unit,
    power_linear_unit,
)
from .relu_variants import blrelu, dual_line, lrelu, mrelu, relu, rrelu, trec
from .sigmoid_tanh_variants import aria2, isru, sqnl, tanh_exp
from .specialized_variants import (
    erf_act,
    hat,
    prelu,
    resp,
    sin_sig,
    suish,
)
from .tanh_variants import penalized_tanh, stanhplus, tanh_linear_unit, tanhsig

__all__ = [
    # ReLU variants
    "relu",
    "lrelu",
    "blrelu",
    "rrelu",
    "trec",
    "dual_line",
    "mrelu",
    # Sigmoid/Tanh variants
    "sqnl",
    "tanh_exp",
    "isru",
    # Polynomial/Power variants
    "polynomial_linear_unit",
    "power_function_linear_unit",
    "power_linear_unit",
    "inverse_polynomial_linear_unit",
    # Log/Exp/Softplus variants
    "loglog",
    "loglogish",
    "logish",
    "soft_exponential",
    "softplus_linear_unit",
    # ELU variants
    "abslu",
    "celu",
    "elu",
    "selu",
    "isrlu",
    # GELU/Swish variants
    "gelu",
    "hard_sigmoid",
    "hard_swish",
    "mish",
    "phish",
    "silu",
    # Tanh variants
    "penalized_tanh",
    "stanhplus",
    "tanh_linear_unit",
    "tanhsig",
    # Sigmoid variants
    isrlu,
    # Specialized variants
    "erf_act",
    "hat",
    "prelu",
    "resp",
    "sin_sig",
    "suish",
    "aria2",
]
