"""Public entry points wired up on top of the internal modules.

These are the functions re-exported by ``feret/__init__.py``. They exist
purely to preserve the historical signatures and to compose the internal
helpers into the same call sequence the legacy ``Calculater`` had.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import _diameters, _hull
from ._result import FeretResult


def _build_result(img: NDArray[np.number], edge: bool) -> FeretResult:
    hull = _hull.extract_hull(img, edge)
    y0, x0 = _hull.hull_pseudo_center(hull)
    return FeretResult(img=img.astype(float), hull=hull, edge=edge, y0=y0, x0=x0)


def _fill_min(res: FeretResult) -> None:
    res.minf, res.minf_coords, res.minf_angle, res.minf_t = _diameters.min_feret(res.hull, res.edge)


def _fill_max(res: FeretResult) -> None:
    res.maxf, res.maxf_coords, res.maxf_angle, res.maxf_t = _diameters.max_feret(res.hull, res.edge)


def _fill_min90(res: FeretResult) -> None:
    value, coords, angle, t = _diameters.perpendicular_feret(
        res.hull, res.y0, res.x0, res.minf_angle, res.edge
    )
    res.minf90, res.minf90_coords, res.minf90_angle, res.minf90_t = (
        value,
        coords,
        angle,
        t,
    )


def _fill_max90(res: FeretResult) -> None:
    value, coords, angle, t = _diameters.perpendicular_feret(
        res.hull, res.y0, res.x0, res.maxf_angle, res.edge
    )
    res.maxf90, res.maxf90_coords, res.maxf90_angle, res.maxf90_t = (
        value,
        coords,
        angle,
        t,
    )


# ---------------------------------------------------------------------------
# Public API — signatures must remain byte-identical to the pre-refactor
# module.
# ---------------------------------------------------------------------------


def calc(img: NDArray[np.number], edge: bool = False) -> FeretResult:
    """Compute every Feret diameter and return the full result object.

    Parameters
    ----------
    img
        2-D binary image. Any non-zero value is treated as foreground.
    edge
        If ``True``, use pixel edges instead of pixel centers (matches the
        ImageJ Feret convention).

    Returns
    -------
    FeretResult
        Object exposing ``maxf``, ``minf``, ``minf90``, ``maxf90`` and the
        corresponding ``*_angle`` attributes.
    """

    res = _build_result(img, edge)
    _fill_min(res)
    _fill_min90(res)
    _fill_max(res)
    _fill_max90(res)
    return res


def all(  # noqa: A001 - shadow of builtin kept for API back-compat
    img: NDArray[np.number], edge: bool = False
) -> tuple[float, float, float, float]:
    """Return ``(maxf, minf, minf90, maxf90)`` in that order."""

    res = calc(img, edge)
    return res.maxf, res.minf, res.minf90, res.maxf90


def plot(img: NDArray[np.number], edge: bool = False) -> None:
    """Compute all diameters and render them with matplotlib.

    Matplotlib is only imported inside this call, so importing ``feret``
    never pulls it in.
    """

    from ._plotting import plot_result

    plot_result(calc(img, edge))


def max(  # noqa: A001 - shadow of builtin kept for API back-compat
    img: NDArray[np.number], edge: bool = False
) -> float:
    """Return only the maximum Feret diameter."""

    res = _build_result(img, edge)
    _fill_max(res)
    return res.maxf


def min(  # noqa: A001 - shadow of builtin kept for API back-compat
    img: NDArray[np.number], edge: bool = False
) -> float:
    """Return only the minimum Feret diameter."""

    res = _build_result(img, edge)
    _fill_min(res)
    return res.minf


def min90(img: NDArray[np.number], edge: bool = False) -> float:
    """Return the Feret diameter 90° to the minimum Feret diameter."""

    res = _build_result(img, edge)
    _fill_min(res)
    _fill_min90(res)
    return res.minf90


def max90(img: NDArray[np.number], edge: bool = False) -> float:
    """Return the Feret diameter 90° to the maximum Feret diameter."""

    res = _build_result(img, edge)
    _fill_max(res)
    _fill_max90(res)
    return res.maxf90
