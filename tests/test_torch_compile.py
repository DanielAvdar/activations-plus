import os
import sys
from unittest.mock import patch

import pytest
import torch
from torch import Tensor

from activations_plus import ELiSH, HardSwish
from activations_plus.bent_identity.bent_identity_func import BentIdentity
from activations_plus.entmax.entmax import Entmax
from activations_plus.soft_clipping.soft_clipping_func import SoftClipping
from activations_plus.sparsemax import Sparsemax
from activations_plus.srelu.srelu_func import SReLU

compile_backends = []

if sys.platform.startswith("linux"):
    compile_backends += [
        "inductor",
    ]
if torch.cuda.is_available():
    compile_backends += ["cudagraphs"]


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        ((2, 5), -1),
        ((3, 10), -1),
        ((4, 6, 10), 1),
        ((1, 20), 0),
        ((3, 4, 5, 6), 2),
    ],
)
@pytest.mark.parametrize("backend", compile_backends)
@pytest.mark.parametrize("activation", [Sparsemax, Entmax, BentIdentity, ELiSH, HardSwish, SoftClipping, SReLU])
def test_activations_torch_compile(input_shape, dim, backend, activation):
    with patch.dict(os.environ, {"TORCH_LOGS": "+dynamo", "TORCHDYNAMO_VERBOSE": "1"}):
        activation_instance = activation(dim=dim) if hasattr(activation, "dim") else activation()
        input_tensor = torch.randn(input_shape, requires_grad=True)
        activation_instance(input_tensor)

        compiled_activation = torch.compile(activation_instance, backend=backend)
        output: Tensor = compiled_activation(input_tensor)

        # Verify basic properties of output (e.g., shape remains the same)
        assert output.shape == input_shape, "Output shape mismatch."
        assert not torch.any(torch.isnan(output)), "Output contains NaN."

        # Optional: Check grad compatibility
        output.sum().backward()
        assert input_tensor.grad is not None, "Gradients did not compute properly."
