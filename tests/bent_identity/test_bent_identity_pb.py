import torch
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as hnp

from activations_plus.bent_identity.bent_identity_func import BentIdentity


@given(
    hnp.arrays(
        dtype=float,
        shape=hnp.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=100),
        elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
)
def test_bent_identity_randomized(data):
    activation = BentIdentity()
    x = torch.tensor(data, dtype=torch.float32)
    y = activation(x)
    expected = (torch.sqrt(x**2 + 1) - 1) / 2 + x
    assert torch.allclose(y, expected)
