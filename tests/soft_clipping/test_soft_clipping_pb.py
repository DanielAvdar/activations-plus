import hypothesis.extra.numpy as hnp
import torch
from hypothesis import given, strategies as st

from activations_plus.soft_clipping.soft_clipping_func import SoftClipping


@given(
    hnp.arrays(
        dtype=float,
        shape=hnp.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=100),
        elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
)
def test_soft_clipping_randomized(data):
    activation = SoftClipping(x_min=-1.0, x_max=1.0)
    x = torch.tensor(data, dtype=torch.float32)
    y = activation(x)
    expected = -1.0 + (1.0 - (-1.0)) * torch.sigmoid(x)
    assert torch.allclose(y, expected)
