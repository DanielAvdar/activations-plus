"""activations_plus package initialization."""

from importlib.metadata import version

from .bent_identity import BentIdentity
from .elish import ELiSH
from .entmax import Entmax
from .maxout import Maxout
from .soft_clipping import SoftClipping
from .sparsemax import Sparsemax
from .srelu import SReLU

__version__ = version("activations-plus")
__all__ = ["ELiSH", "BentIdentity", "SoftClipping", "Maxout", "SReLU", "Sparsemax", "Entmax"]
