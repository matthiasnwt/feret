"""Feret-diameter algorithms operating on a convex hull.

The functions in this module are numerically identical to the original
``feret.main.Calculater`` methods; they have only been unpacked into free
functions with clear inputs and outputs so the arithmetic can be audited and
tested in isolation. The refactor introduces exactly one intentional
substitution:

* ``np.cross`` on 2-D vectors was removed in NumPy 2.0. We inline its
  equivalent formula (``a[0]*b[1] - a[1]*b[0]``) below. Because that formula
  is identical to what ``np.cross`` used to compute — same operands, same
  order — the result is bit-identical on NumPy 1.x, and the package now works
  on NumPy 2.x as well.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform

FloatArr = NDArray[np.float64]

# ---------------------------------------------------------------------------
# minimum Feret diameter
# ---------------------------------------------------------------------------


def _perp_distances_to_edge(p1: FloatArr, p2: FloatArr, hull: FloatArr) -> FloatArr:
    """Perpendicular distance from every hull point to the line through
    ``p1`` and ``p2``.

    Inlined 2-D cross product replaces the removed ``np.cross`` overload.
    """

    v = p2 - p1  # (2,)
    w = p1 - hull.T  # (N, 2)
    # np.cross([v0, v1], [w0, w1]) == v0*w1 - v1*w0
    cross = v[0] * w[:, 1] - v[1] * w[:, 0]
    return np.abs(cross / norm(v))


def min_feret(hull: FloatArr, edge: bool) -> tuple[float, FloatArr, float, float]:
    """Exact minimum Feret diameter of ``hull``.

    Iterates every edge of the hull. For each edge, treats the edge as one
    caliper contact and finds the hull point farthest from that edge; the
    perpendicular distance to that point is the caliper opening for that
    orientation. The smallest opening over all edges is the minferet.

    Returns
    -------
    minf, minf_coords, minf_angle, minf_t
        Same semantics and units as the legacy
        ``Calculater.calculate_minferet`` attributes.
    """

    length = hull.shape[1]

    Ds = np.empty(length)
    ps = np.empty((length, 3, 2))

    for i in range(length):
        p1 = hull.T[i]
        p2 = hull.T[(i + 1) % length]

        ds = _perp_distances_to_edge(p1, p2, hull)

        Ds[i] = np.max(ds)
        d_i = np.where(ds == Ds[i])[0][0]
        p3 = hull.T[d_i]
        ps[i] = np.array((p1, p2, p3))

    minf = float(np.min(Ds))

    minf_index = np.where(Ds == minf)[0][0]
    (y0, x0), (y1, x1), (y2, x2) = ps[minf_index]

    if x0 == x1:
        minf_angle = 0.0
    else:
        m = (y0 - y1) / (x0 - x1)
        minf_angle = float(np.arctan(m) + np.pi / 2)

    minf_coords = np.array(((y0, x0), (y1, x1), (y2, x2)))

    if minf_angle < 0:
        minf_angle += np.pi

    if edge:
        minf /= 2.0
        minf_coords = minf_coords / 2.0

    if x0 == x1:
        minf_t = float(minf_coords.T[0][2])
    else:
        minf_t = float(minf_coords.T[0][2] - np.tan(minf_angle) * minf_coords.T[1][2])

    return minf, minf_coords, minf_angle, minf_t


# ---------------------------------------------------------------------------
# maximum Feret diameter
# ---------------------------------------------------------------------------


def max_feret(hull: FloatArr, edge: bool) -> tuple[float, FloatArr, float, float]:
    """Exact maximum Feret diameter of ``hull`` (largest pairwise distance).

    Returns
    -------
    maxf, maxf_coords, maxf_angle, maxf_t
        Same semantics as the legacy ``Calculater.calculate_maxferet``
        attributes.
    """

    pdists = pdist(hull.T, "euclidean")

    maxf = float(np.max(pdists))

    maxf_coords_index = np.where(squareform(pdists) == maxf)[0]

    # If more than one pair realises the maxferet, break ties the same way
    # the original code did: split the argmax indices in half and take the
    # first element of each half.
    maxf_coords_index_y = maxf_coords_index[: len(maxf_coords_index) // 2][0]
    maxf_coords_index_x = maxf_coords_index[len(maxf_coords_index) // 2 :][0]

    maxf_coords = hull.T[np.array((maxf_coords_index_x, maxf_coords_index_y))]

    ((y0, x0), (y1, x1)) = maxf_coords

    if x1 == x0:
        maxf_angle = float(np.pi / 2)
    else:
        m = (y0 - y1) / (x0 - x1)
        maxf_angle = float(np.arctan(m))

    if maxf_angle < 0:
        maxf_angle += np.pi

    if edge:
        maxf /= 2.0
        maxf_coords = maxf_coords / 2

    if x1 == x0:
        maxf_t = -np.inf
    else:
        maxf_t = float(maxf_coords.T[0][1] - np.tan(maxf_angle) * maxf_coords.T[1][1])

    return maxf, maxf_coords, maxf_angle, maxf_t


# ---------------------------------------------------------------------------
# 90°-rotated caliper distance
# ---------------------------------------------------------------------------


def caliper_extents_at_angle(
    hull: FloatArr, y0: float, x0: float, angle: float
) -> tuple[float, FloatArr]:
    """Caliper opening along direction ``angle`` (in radians) and the two
    hull points that realise it.

    Note
    ----
    ``y0`` and ``x0`` enter as a constant offset in ``ds`` and therefore
    cancel out under ``argmax`` / ``argmin``. See ``FINDINGS.md``. We keep
    them in the signature so callers still supply the historical values and
    the function stays byte-for-byte compatible with the legacy version.
    """

    m = np.tan(angle)
    ds = np.cos(angle) * (y0 - hull[0]) - np.sin(angle) * (x0 - hull[1])
    max_i = int(np.argmax(ds))
    min_i = int(np.argmin(ds))

    t_max = hull.T[max_i][0] - m * hull.T[max_i][1]
    t_min = hull.T[min_i][0] - m * hull.T[min_i][1]

    max_xy = [hull[0][max_i], hull[1][max_i]]
    min_xy = [hull[0][min_i], hull[1][min_i]]

    distance = float(np.abs(t_max - t_min) / np.sqrt(1 + m**2))
    coords = np.array([max_xy, min_xy])
    return distance, coords


def perpendicular_feret(
    hull: FloatArr,
    y0: float,
    x0: float,
    base_angle: float,
    edge: bool,
) -> tuple[float, FloatArr, float, float]:
    """Feret diameter perpendicular to ``base_angle``.

    ``base_angle`` is the angle of the reference diameter (e.g. ``minf_angle``
    or ``maxf_angle``). The perpendicular caliper is at ``base_angle +
    pi/2``.

    Returns
    -------
    value, coords, angle, t
        ``value`` is the caliper opening (halved when ``edge`` is set),
        ``coords`` are the two touching hull points, ``angle`` is
        ``base_angle + pi/2``, and ``t`` reproduces the legacy line-intercept
        conventions used by the plotter.
    """

    perp_angle = base_angle + np.pi / 2
    value, coords = caliper_extents_at_angle(hull, y0, x0, perp_angle - np.pi / 2)

    if edge:
        value /= 2
        coords = coords / 2

    if perp_angle == 0 or perp_angle == np.pi:
        t = float(coords.T[0][0])
    elif perp_angle == np.pi / 2:
        t = np.inf
    else:
        t = float(coords.T[0][1] - np.tan(perp_angle) * coords.T[1][1])

    return value, coords, float(perp_angle), t
