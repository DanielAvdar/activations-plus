import torch
from hypothesis import given, strategies as st

from activations_plus.soft_clipping.soft_clipping_func import SoftClipping


@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=100)
)
def test_soft_clipping_randomized(data):
    activation = SoftClipping(x_min=-1.0, x_max=1.0)
    x = torch.tensor(data, dtype=torch.float32)
    y = activation(x)
    expected = -1.0 + (1.0 - (-1.0)) * torch.sigmoid(x)
    assert torch.allclose(y, expected)
