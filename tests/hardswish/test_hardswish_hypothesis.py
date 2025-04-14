import torch
from hypothesis import given, strategies as st

from activations_plus.hardswish.hardswish_func import HardSwish


@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=100)
)
def test_hardswish_randomized(data):
    activation = HardSwish()
    x = torch.tensor(data, dtype=torch.float32)
    y = activation(x)
    expected = x * torch.clamp((x + 3) / 6, min=0, max=1)
    assert torch.allclose(y, expected)
