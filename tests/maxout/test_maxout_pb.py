import torch
from hypothesis import given, strategies as st

from activations_plus.maxout.maxout_func import Maxout


@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=2, max_size=100)
)
def test_maxout_randomized(data):
    activation = Maxout(num_pieces=2)
    # Ensure the data size is even
    if len(data) % 2 != 0:
        data = data[:-1]  # Truncate the last element if the size is odd
    x = torch.tensor(data, dtype=torch.float32).view(-1, 2)
    y = activation(x)
    expected = torch.max(x.view(x.size(0), 1, 2), dim=-1)[0]
    assert torch.allclose(y, expected)
