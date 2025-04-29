"""activations_plus package initialization."""

from importlib.metadata import version

from .entmax import Entmax
from .maxout import Maxout
from .soft_clipping import SoftClipping
from .sparsemax import Sparsemax

__version__ = version("activations-plus")
__all__ = ["SoftClipping", "Maxout", "Sparsemax", "Entmax"]
