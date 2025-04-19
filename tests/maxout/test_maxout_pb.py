import hypothesis.extra.numpy as hnp
import torch
from hypothesis import given, strategies as st

from activations_plus.maxout.maxout_func import Maxout


@given(
    hnp.arrays(
        dtype=float,
        shape=hnp.array_shapes(min_dims=1, max_dims=1, min_side=2, max_side=100),
        elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
)
def test_maxout_randomized(data):
    activation = Maxout(num_pieces=2)
    # Ensure the data size is even
    if data.shape[0] % 2 != 0:
        data = data[:-1]  # Truncate the last element if the size is odd
    x = torch.tensor(data, dtype=torch.float32).view(-1, 2)
    y = activation(x)
    expected = torch.max(x.view(x.size(0), 1, 2), dim=-1)[0]
    assert torch.allclose(y, expected)
