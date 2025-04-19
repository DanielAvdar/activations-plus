import hypothesis.extra.numpy as hnp
import torch
from hypothesis import given, strategies as st

from activations_plus.elish.elish_func import ELiSH


@given(
    hnp.arrays(
        dtype=float,
        shape=hnp.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=100),
        elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
)
def test_elish_randomized(data):
    activation = ELiSH()
    x = torch.tensor(data, dtype=torch.float32)
    y = activation(x)
    expected_positive = x / (1 + torch.exp(-x))
    expected_negative = torch.exp(x) - 1
    assert torch.allclose(y[x >= 0], expected_positive[x >= 0])
    assert torch.allclose(y[x < 0], expected_negative[x < 0])
