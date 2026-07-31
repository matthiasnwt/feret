"""feret — compute the Feret diameters of a binary image.

Public API (unchanged from previous versions):

* :func:`calc`
* :func:`all`
* :func:`plot`
* :func:`max`
* :func:`min`
* :func:`min90`
* :func:`max90`

All entry points accept a 2-D NumPy array where any non-zero value is
foreground, and an optional ``edge`` keyword that switches between the pixel-
center convention (default) and the pixel-edge convention (matches ImageJ).
"""

from __future__ import annotations

from ._api import all, calc, max, max90, min, min90, plot  # noqa: A004,F401
from ._version import __version__

__all__ = [
    "__version__",
    "all",
    "calc",
    "max",
    "max90",
    "min",
    "min90",
    "plot",
]
