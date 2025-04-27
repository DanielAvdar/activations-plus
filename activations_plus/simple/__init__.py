"""Simple activation functions and their variants for neural networks."""

from .log_exp_softplus_variants import logish, loglog, loglogish, soft_exponential, softplus_linear_unit
from .polynomial_power_variants import (
    inverse_polynomial_linear_unit,
    polynomial_linear_unit,
    power_function_linear_unit,
    power_linear_unit,
)
from .relu_variants import blrelu, dual_line, lrelu, mrelu, relu, rrelu, trec
from .sigmoid_tanh_variants import hardtanh, sigmoid, softplus, softsign, sqnl, tanh, tanh_exp

__all__ = [
    "relu",
    "lrelu",
    "blrelu",
    "rrelu",
    "trec",
    "dual_line",
    "mrelu",
    "sigmoid",
    "tanh",
    "hardtanh",
    "softsign",
    "sqnl",
    "softplus",
    "tanh_exp",
    "polynomial_linear_unit",
    "power_function_linear_unit",
    "power_linear_unit",
    "inverse_polynomial_linear_unit",
    "loglog",
    "loglogish",
    "logish",
    "soft_exponential",
    "softplus_linear_unit",
]
