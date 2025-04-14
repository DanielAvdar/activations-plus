from importlib.metadata import version

from .bent_identity import BentIdentity
from .elish import ELiSH
from .hardswish import HardSwish
from .maxout import Maxout
from .soft_clipping import SoftClipping

__version__ = version("activations-plus")
__all__ = ["ELiSH", "HardSwish", "BentIdentity", "SoftClipping", "Maxout"]
