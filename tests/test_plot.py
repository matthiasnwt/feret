"""Smoke tests for ``feret.plot``.

We cannot compare pixels of a matplotlib figure, so we verify:

* ``feret.plot`` runs to completion without raising for a range of shapes,
* it runs both with ``edge=False`` and ``edge=True``,
* it registers exactly one new matplotlib figure per call,
* it does not require an interactive backend (conftest sets Agg).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
from shapes import SHAPES

import feret

_SMOKE_SHAPES = [
    "square",
    "rectangle",
    "rotated_square",
    "circle",
    "ellipse",
    "l_shape",
    "concave_c_shape",
    "cross_shape",
    "touching_border",
    "random_blob_a",
    "repo_fixture",
]


@pytest.fixture(autouse=True)
def _close_all_figures():
    plt.close("all")
    yield
    plt.close("all")


@pytest.mark.parametrize("shape_name", _SMOKE_SHAPES)
@pytest.mark.parametrize("edge", [False, True], ids=["edge_false", "edge_true"])
def test_plot_runs_without_error(shape_name: str, edge: bool) -> None:
    img = SHAPES[shape_name]()
    before = len(plt.get_fignums())
    feret.plot(img, edge=edge)
    after = len(plt.get_fignums())
    assert after - before == 1, "expected feret.plot to open exactly one figure"


def test_plot_default_edge_is_false() -> None:
    img = SHAPES["square"]()
    feret.plot(img)
    assert plt.get_fignums(), "plot() should have created a figure"
