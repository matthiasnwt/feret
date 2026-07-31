"""Golden-master characterization tests for the ``feret`` public API.

Every documented public entry point is exercised against a fixed set of
synthetic binary shapes and its output is compared to values captured from the
current implementation (see ``tests/generate_golden.py``). The tests exist to
lock behaviour down so that the internal refactor cannot silently change any
numerical result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from shapes import SHAPES

import feret

GOLDEN_PATH = Path(__file__).with_name("golden.json")
GOLDEN: dict[str, dict[str, dict[str, Any]]] = json.loads(GOLDEN_PATH.read_text())

# Attributes on ``feret.calc(...)`` result that the README documents.
PUBLIC_ATTRS = (
    "maxf",
    "minf",
    "minf90",
    "maxf90",
    "maxf_angle",
    "minf_angle",
    "maxf90_angle",
    "minf90_angle",
)

# Exact equality is expected — the refactor must not change a single bit of the
# computed numbers. ``assert_allclose`` with ``rtol=0`` and ``atol=0`` is the
# strictest possible tolerance.
_RTOL = 0.0
_ATOL = 0.0


@pytest.fixture(params=sorted(SHAPES.keys()))
def shape_name(request) -> str:
    return request.param


@pytest.fixture
def image(shape_name: str) -> np.ndarray:
    return SHAPES[shape_name]()


@pytest.fixture(params=[False, True], ids=["edge_false", "edge_true"])
def edge(request) -> bool:
    return request.param


@pytest.fixture
def golden(shape_name: str, edge: bool) -> dict[str, Any]:
    key = "edge_true" if edge else "edge_false"
    return GOLDEN[shape_name][key]


def _assert_close(actual: float, expected: float, label: str) -> None:
    np.testing.assert_allclose(actual, expected, rtol=_RTOL, atol=_ATOL, err_msg=label)


class TestCalcResult:
    """``feret.calc`` returns an object with the documented attributes."""

    def test_all_documented_attributes_match_golden(
        self, image: np.ndarray, edge: bool, golden: dict[str, Any]
    ) -> None:
        res = feret.calc(image, edge=edge)
        for attr in PUBLIC_ATTRS:
            _assert_close(getattr(res, attr), golden[attr], attr)

    def test_default_edge_kwarg_is_false(self, image: np.ndarray) -> None:
        res_default = feret.calc(image)
        res_explicit = feret.calc(image, edge=False)
        for attr in PUBLIC_ATTRS:
            _assert_close(getattr(res_default, attr), getattr(res_explicit, attr), attr)


class TestAll:
    """``feret.all`` returns ``(maxf, minf, minf90, maxf90)`` in that order."""

    def test_returns_four_values_in_documented_order(
        self, image: np.ndarray, edge: bool, golden: dict[str, Any]
    ) -> None:
        maxf, minf, minf90, maxf90 = feret.all(image, edge=edge)
        _assert_close(maxf, golden["maxf"], "all[0] maxf")
        _assert_close(minf, golden["minf"], "all[1] minf")
        _assert_close(minf90, golden["minf90"], "all[2] minf90")
        _assert_close(maxf90, golden["maxf90"], "all[3] maxf90")

    def test_default_edge_kwarg_is_false(self, image: np.ndarray) -> None:
        assert feret.all(image) == feret.all(image, edge=False)


class TestScalarHelpers:
    def test_max(self, image, edge, golden) -> None:
        _assert_close(feret.max(image, edge=edge), golden["maxf"], "feret.max")

    def test_min(self, image, edge, golden) -> None:
        _assert_close(feret.min(image, edge=edge), golden["minf"], "feret.min")

    def test_min90(self, image, edge, golden) -> None:
        _assert_close(feret.min90(image, edge=edge), golden["minf90"], "feret.min90")

    def test_max90(self, image, edge, golden) -> None:
        _assert_close(feret.max90(image, edge=edge), golden["maxf90"], "feret.max90")

    def test_default_edge_kwarg_is_false(self, image) -> None:
        assert feret.max(image) == feret.max(image, edge=False)
        assert feret.min(image) == feret.min(image, edge=False)
        assert feret.min90(image) == feret.min90(image, edge=False)
        assert feret.max90(image) == feret.max90(image, edge=False)


class TestReturnTypes:
    """Return types must not change: ``calc`` -> object, ``all`` -> tuple,
    scalar helpers -> Python or NumPy scalar (must be finite)."""

    def test_calc_returns_object_with_all_attrs(self, image) -> None:
        res = feret.calc(image)
        for attr in PUBLIC_ATTRS:
            assert hasattr(res, attr), f"missing attribute {attr!r}"

    def test_all_returns_tuple_of_length_four(self, image) -> None:
        out = feret.all(image)
        assert isinstance(out, tuple)
        assert len(out) == 4

    def test_scalar_helpers_return_finite_number(self, image) -> None:
        for fn in (feret.max, feret.min, feret.min90, feret.max90):
            value = fn(image)
            assert np.isfinite(value), f"{fn.__name__} returned non-finite {value!r}"
