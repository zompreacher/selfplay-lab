"""A measured fly connectome used as a fixed reservoir for self-play agents.

The connectome is not a policy and is not trained. It is a recurrent nonlinear
map whose weights were measured by electron microscopy rather than by gradient
descent. The agent's trainable parameters sit entirely downstream of it.

The only claim this package tries to support is the one its null models test:
whether the *measured* wiring does anything a degree-matched rewiring of the
same wiring does not. See `nulls.py`.
"""

from .extract import CircuitExtract, load_extract, signed_matrix
from .nulls import NULL_MODELS, apply_null
from .reservoir import FlyReservoir

__all__ = [
    "NULL_MODELS",
    "CircuitExtract",
    "FlyReservoir",
    "apply_null",
    "load_extract",
    "signed_matrix",
]
