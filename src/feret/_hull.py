"""Convex-hull extraction for a binary image.

The convex hull captures every pixel of the object that can possibly touch a
caliper, which is why every Feret computation in this package operates on the
hull rather than on the raw pixel mask. Two conventions are supported:

* pixel *centers* (default) — a point per foreground pixel, at its integer
  coordinate;
* pixel *edges* — nine points per foreground pixel covering its corners and
  midpoints, so the resulting Feret diameters match the ImageJ convention.

Both branches return an ``(2, N)`` ``float64`` array with rows ``[y, x]``.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np
from numpy.typing import NDArray


def extract_hull(img: NDArray[np.number], edge: bool) -> NDArray[np.float64]:
    """Return the convex hull of the foreground of ``img`` as ``(2, N)``.

    Parameters
    ----------
    img
        2-D array. Any non-zero value is treated as foreground.
    edge
        If ``True`` expand each foreground pixel to its nine cover points and
        double the resulting coordinates so integer arithmetic in OpenCV stays
        exact; the caller is responsible for halving the diameters and
        coordinates it derives from the hull.

    Notes
    -----
    The float-image cast (``img.astype(float)``) mirrors the original
    implementation and is what allowed dtype-tolerant inputs. It is preserved
    here bit-for-bit so nothing downstream sees a different value.
    """

    img_f = img.astype(float)

    if edge:
        ys, xs = np.nonzero(img_f)
        new_xs = np.concatenate(
            (xs + 0.5, xs + 0.5, xs - 0.5, xs - 0.5, xs, xs + 0.5, xs - 0.5, xs, xs)
        )
        new_ys = np.concatenate(
            (ys + 0.5, ys - 0.5, ys + 0.5, ys - 0.5, ys, ys, ys, ys + 0.5, ys - 0.5)
        )
        new_ys = (new_ys * 2).astype(int)
        new_xs = (new_xs * 2).astype(int)
        pts = np.array([new_ys, new_xs]).T
    else:
        pts = np.transpose(np.nonzero(img_f))

    hull = cv.convexHull(pts).T.reshape(2, -1).astype(float)
    return hull


def hull_pseudo_center(hull: NDArray[np.float64]) -> tuple[float, float]:
    """Reproduce the original ``ndimage.center_of_mass(self.hull)`` value.

    This is preserved verbatim because it is *observable* state on the legacy
    ``Calculater`` object. See ``FINDINGS.md`` — the value is mathematically
    nonsensical, but is used only as a constant offset inside
    :func:`caliper_extents_at_angle`, where ``argmax`` / ``argmin`` cancel any
    constant offset. Every numerical result of the public API is therefore
    unaffected by this quantity, but we still compute it identically so that
    any code inspecting internal attributes sees exactly the historical value.
    """

    from scipy import ndimage  # local import so plotting extras stay optional

    y0, x0 = ndimage.center_of_mass(hull)
    return float(y0), float(x0)
