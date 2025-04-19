import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from activations_plus.bent_identity import BentIdentity
from activations_plus.elish import ELiSH
from activations_plus.entmax import Entmax
from activations_plus.maxout import Maxout
from activations_plus.soft_clipping import SoftClipping
from activations_plus.sparsemax import Sparsemax
from activations_plus.srelu import SReLU

activation_params = [
    (Sparsemax, {"dim": -1}),
    (Entmax, {"dim": -1}),
    (BentIdentity, {}),
    (ELiSH, {}),
    (Maxout, {"num_pieces": 2}),
    (SoftClipping, {}),
    (SReLU, {}),
]
input_shapes = [
    (4, 5),
    (4, 5),
    (4, 5),
    (4, 5),
    (4, 6),
    (4, 5),
    (4, 5),
]


def _all_devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


@pytest.mark.parametrize("activation_info,input_shape", zip(activation_params, input_shapes))
@pytest.mark.parametrize("device", _all_devices())
def test_activation_grad_check(activation_info, input_shape, device):
    activation_cls, kwargs = activation_info
    activation = activation_cls(**kwargs)
    x = torch.randn(*input_shape, dtype=torch.double, requires_grad=True, device=device)
    assert gradcheck(activation, (x,), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("activation_info,input_shape", zip(activation_params, input_shapes))
@pytest.mark.parametrize("device", _all_devices())
def test_activation_gradgrad_check(activation_info, input_shape, device):
    activation_cls, kwargs = activation_info
    activation = activation_cls(**kwargs)
    x = torch.randn(*input_shape, dtype=torch.double, requires_grad=True, device=device)
    assert gradgradcheck(activation, (x,), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("activation_info,input_shape", zip(activation_params, input_shapes))
def test_activation_empty_and_singleton(activation_info, input_shape):
    activation_cls, kwargs = activation_info
    activation = activation_cls(**kwargs)
    # Empty tensor
    x_empty = torch.empty(0, *input_shape[1:], dtype=torch.double, requires_grad=True)
    y_empty = activation(x_empty)
    assert y_empty.shape == x_empty.shape
    # Singleton tensor
    x_single = torch.ones(1, *input_shape[1:], dtype=torch.double, requires_grad=True)
    y_single = activation(x_single)
    assert y_single.shape == x_single.shape


@pytest.mark.parametrize("activation_info,input_shape", zip(activation_params, input_shapes))
def test_activation_dtype_consistency(activation_info, input_shape):
    activation_cls, kwargs = activation_info
    activation = activation_cls(**kwargs)
    for dtype in [torch.float16, torch.float32, torch.float64]:
        x = torch.randn(*input_shape, dtype=dtype, requires_grad=True)
        y = activation(x)
        assert y.dtype == x.dtype


@pytest.mark.parametrize("activation_info,input_shape", zip(activation_params, input_shapes))
def test_activation_broadcasting(activation_info, input_shape):
    activation_cls, kwargs = activation_info
    activation = activation_cls(**kwargs)
    x = torch.randn(*input_shape, dtype=torch.double, requires_grad=True)
    # Broadcast with extra dimension
    y = activation(x + torch.randn(1, *input_shape[1:], dtype=torch.double))
    assert y.shape == x.shape
