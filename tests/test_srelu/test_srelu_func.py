# Enhanced tests for the SReLU (S-shaped ReLU) function
import torch
from hypothesis import given, strategies as st

from activations_plus.srelu.srelu_func import SReLU


def test_srelu_basic():
    srelu = SReLU(lower_threshold=-1.0, upper_threshold=1.0)
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = srelu(x)
    assert torch.allclose(y, torch.tensor([-1.0, -1.0, 0.0, 1.0, 1.0]))


def test_srelu_custom_thresholds():
    srelu = SReLU(lower_threshold=-0.5, upper_threshold=0.5)
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    y = srelu(x)
    assert torch.allclose(y, torch.tensor([-0.5, -0.5, 0.0, 0.5, 0.5]))


def test_srelu_invalid_thresholds():
    try:
        SReLU(lower_threshold=1.0, upper_threshold=-1.0)
    except ValueError as e:
        assert str(e) == "lower_threshold must be less than or equal to upper_threshold"


def test_srelu_large_tensor():
    srelu = SReLU(lower_threshold=-1.0, upper_threshold=1.0)
    x = torch.randn(1000, 1000)
    y = srelu(x)
    assert y.shape == x.shape
    assert torch.all(y >= -1.0) and torch.all(y <= 1.0)


@given(
    random_data=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=2, max_size=100
    ),
    lower_threshold=st.floats(min_value=-1e3, max_value=0),
    upper_threshold=st.floats(min_value=0, max_value=1e3),
)
def test_srelu_randomized(random_data, lower_threshold, upper_threshold):
    srelu = SReLU(lower_threshold=lower_threshold, upper_threshold=upper_threshold)
    x = torch.tensor(random_data, dtype=torch.float32)
    y = srelu(x)
    assert y.shape == x.shape
    assert torch.all(y >= lower_threshold) and torch.all(y <= upper_threshold)
