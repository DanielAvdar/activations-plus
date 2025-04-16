import torch
from hypothesis import given, strategies as st

from activations_plus.bent_identity.bent_identity_func import BentIdentity


@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=100)
)
def test_bent_identity_randomized(data):
    activation = BentIdentity()
    x = torch.tensor(data, dtype=torch.float32)
    y = activation(x)
    expected = (torch.sqrt(x**2 + 1) - 1) / 2 + x
    assert torch.allclose(y, expected)
