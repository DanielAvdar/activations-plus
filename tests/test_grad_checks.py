import pytest
import torch
from torch.autograd import gradcheck

from activations_plus.sparsemax import Sparsemax
from activations_plus.entmax import Entmax
from activations_plus.bent_identity import BentIdentity
from activations_plus.elish import ELiSH
from activations_plus.maxout import Maxout
from activations_plus.soft_clipping import SoftClipping
from activations_plus.srelu import SReLU

activation_params = [
    (Sparsemax, {'dim': -1}),
    (Entmax, {'dim': -1}),
    (BentIdentity, {}),
    (ELiSH, {}),
    (Maxout, {'num_pieces': 2}),
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

@pytest.mark.parametrize("activation_info,input_shape", zip(activation_params, input_shapes))
def test_activation_grad_check(activation_info, input_shape):
    activation_cls, kwargs = activation_info
    activation = activation_cls(**kwargs)
    x = torch.randn(*input_shape, dtype=torch.double, requires_grad=True)
    assert gradcheck(activation, (x,), eps=1e-6, atol=1e-4)
