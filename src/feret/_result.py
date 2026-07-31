"""The result object returned by :func:`feret.calc`.

Public attributes documented in the README:

* ``maxf``, ``minf``, ``minf90``, ``maxf90``
* ``maxf_angle``, ``minf_angle``, ``maxf90_angle``, ``minf90_angle``

Internal attributes (unchanged from the pre-refactor implementation, still
consumed by :mod:`feret._plotting`):

* ``img``, ``hull``, ``edge``, ``y0``, ``x0``
* ``{maxf,minf,maxf90,minf90}_coords``, ``{maxf,minf,maxf90,minf90}_t``

Attributes are set incrementally as the various compute functions run so that
partial calls (``feret.max``, ``feret.min``, ...) do not pay the cost of
computing what they do not return, but any attribute that was not populated is
simply absent — matching the legacy behaviour where such an attribute would
raise ``AttributeError``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class FeretResult:
    """Container for the four Feret diameters and their supporting geometry.

    The attribute list matches the legacy ``feret.main.Calculater`` object one
    for one, so any external code that accessed those attributes still works.
    New in this release: attributes are declared up front (with type hints)
    for editor discoverability. They stay bare Python floats / NumPy arrays
    to preserve numerical behaviour exactly.
    """

    # Configuration and cached geometry
    img: NDArray[np.floating]
    hull: NDArray[np.float64]
    edge: bool
    y0: float
    x0: float

    # Public numerical results (populated on demand)
    maxf: float
    minf: float
    minf90: float
    maxf90: float

    maxf_angle: float
    minf_angle: float
    maxf90_angle: float
    minf90_angle: float

    # Internal, used by plotting
    maxf_coords: NDArray[np.float64]
    minf_coords: NDArray[np.float64]
    maxf90_coords: NDArray[np.float64]
    minf90_coords: NDArray[np.float64]

    maxf_t: float
    minf_t: float
    maxf90_t: float
    minf90_t: float

    def __init__(
        self,
        img: NDArray[np.floating],
        hull: NDArray[np.float64],
        edge: bool,
        y0: float,
        x0: float,
    ) -> None:
        self.img = img
        self.hull = hull
        self.edge = edge
        self.y0 = y0
        self.x0 = x0

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        parts = []
        for name in ("maxf", "minf", "minf90", "maxf90"):
            value: float | None = getattr(self, name, None)
            parts.append(f"{name}={value!r}")
        return f"FeretResult({', '.join(parts)})"


# Backward-compatibility alias: the legacy class name for anyone who imported
# ``from feret.main import Calculater`` and used ``isinstance`` on the return
# value of ``feret.calc``.
Calculater = FeretResult
