import os
import sys
from unittest.mock import patch

import pytest
import torch
from torch import Tensor

from activations_plus.entmax.entmax import Entmax
from activations_plus.sparsemax import Sparsemax

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
@pytest.mark.parametrize("activation", [Sparsemax, Entmax])
def test_sparsemax_torch_compile(input_shape, dim, backend, activation):
    with patch.dict(os.environ, {"TORCH_LOGS": "+dynamo", "TORCHDYNAMO_VERBOSE": "1"}):
        sparsemax = activation(dim=dim)
        input_tensor = torch.randn(input_shape, requires_grad=True)
        sparsemax(input_tensor)

        compiled_sparsemax = torch.compile(sparsemax, backend=backend)
        output: Tensor = compiled_sparsemax(input_tensor)

        # Verify basic properties of output (e.g., shape remains the same)
        assert output.shape == input_shape, "Output shape mismatch."
        assert not torch.any(torch.isnan(output)), "Output contains NaN."

        # Optional: Check grad compatibility
        output.sum().backward()
        assert input_tensor.grad is not None, "Gradients did not compute properly."
