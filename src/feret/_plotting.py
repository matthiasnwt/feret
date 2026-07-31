"""Matplotlib rendering of a :class:`~feret._result.FeretResult`.

Isolated in its own module so matplotlib is only imported when plotting is
actually requested. All plotting behaviour (colours, labels, axis flip,
figure size, etc.) is preserved exactly as it was in the legacy
``Calculater.plot`` method.
"""

from __future__ import annotations

import numpy as np

from ._result import FeretResult


def _draw_lines(
    ax,
    xs,
    name: str,
    coord1,
    coord2,
    color: str,
    angle: float,
    t: float,
    coords,
    marker: str,
) -> None:
    """Draw the two baseline lines and the through-line for one diameter."""

    ax.scatter(coords.T[1], coords.T[0], c=color, label=name + " coordinates", marker=marker)

    if angle == np.pi / 2:
        ax.axhline(coord1[0], linestyle="--", color=color, label=name + " Baseline")
        ax.axhline(coord2[0], linestyle="--", color=color)
        ax.axvline(coord1[1], color=color, label=name + " Line")
    elif angle == 0:
        ax.axvline(coord1[1], linestyle="--", color=color, label=name + " Baseline")
        ax.axvline(coord2[1], linestyle="--", color=color)
        ax.axhline(coord1[0], color=color, label=name + " Line")
    else:
        base_m = np.tan(angle + np.pi / 2)
        base_t = coord1[0] - base_m * coord1[1]
        anker_t = coord2[0] - base_m * coord2[1]

        base_ys = base_m * xs + base_t
        anker_ys = base_m * xs + anker_t
        ys = np.tan(angle) * xs + t

        ax.plot(xs, base_ys, linestyle="--", color=color, label=name + " Baseline")
        ax.plot(xs, anker_ys, linestyle="--", color=color)
        ax.plot(xs, ys, color=color, label=name + " Line")


def plot_result(result: FeretResult) -> None:
    """Render ``result`` in a new matplotlib figure and show it.

    Preserves the legacy plot layout exactly: title contains
    ``MinFeret`` and ``MaxFeret`` to six decimal places, the image is drawn
    with ``origin='lower'``, and four coloured pairs of lines mark
    MinFeret (red), MaxFeret (blue), MaxFeret90 (green) and MinFeret90
    (orange).
    """

    import matplotlib.pyplot as plt

    fig = plt.figure(dpi=100, figsize=(10, 10))
    ax = fig.gca()
    ax.set_title(f"MinFeret: {result.minf:.6f} ||| MaxFeret: {result.maxf:.6f}")
    ax.imshow(result.img, origin="lower")

    ymax, xmax = result.img.shape
    xs = np.linspace(0, xmax, 2)

    _draw_lines(
        ax,
        xs,
        "MinFeret",
        result.minf_coords[0],
        result.minf_coords[2],
        "red",
        result.minf_angle,
        result.minf_t,
        result.minf_coords,
        "o",
    )
    _draw_lines(
        ax,
        xs,
        "MaxFeret",
        result.maxf_coords[0],
        result.maxf_coords[1],
        "blue",
        result.maxf_angle,
        result.maxf_t,
        result.maxf_coords,
        "o",
    )
    _draw_lines(
        ax,
        xs,
        "MaxFeret90",
        result.maxf90_coords[0],
        result.maxf90_coords[1],
        "green",
        result.maxf90_angle,
        result.maxf90_t,
        result.maxf90_coords,
        "o",
    )
    _draw_lines(
        ax,
        xs,
        "MinFeret90",
        result.minf90_coords[0],
        result.minf90_coords[1],
        "orange",
        result.minf90_angle,
        result.minf90_t,
        result.minf90_coords,
        "o",
    )

    ax.set_ylim(0, ymax)
    ax.set_xlim(0, xmax)
    ax.legend()
    plt.show()
