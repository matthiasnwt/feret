"""Characterizes input-shape and dtype behaviour that is not part of the
happy path.

These tests document the *current* behaviour, warts and all, and are
intentionally lenient about *which* exception type is raised for degenerate
inputs so a refactor to a cleaner internal API stays free to pick a better
exception class without changing observable results for normal inputs. They
still assert that degenerate inputs *fail loudly* rather than silently
returning garbage.
"""

from __future__ import annotations

import numpy as np
import pytest

import feret


def _square_image(dtype, foreground_value: int = 1) -> np.ndarray:
    img = np.zeros((10, 10), dtype=dtype)
    img[2:8, 2:8] = foreground_value
    return img


@pytest.mark.parametrize(
    "dtype",
    [np.uint8, np.uint16, np.int32, np.int64, np.float32, np.float64, bool],
)
def test_dtypes_produce_same_result(dtype) -> None:
    """Every numeric dtype with non-zero foreground pixels must yield the
    same feret values as the ``uint8`` reference."""

    ref = feret.all(_square_image(np.uint8))
    other = feret.all(_square_image(dtype))
    np.testing.assert_allclose(other, ref, rtol=0, atol=0)


@pytest.mark.parametrize("foreground_value", [1, 2, 7, 255])
def test_nonbinary_foreground_values_are_treated_as_true(foreground_value) -> None:
    """The README documents that the object may have any non-zero value."""

    ref = feret.all(_square_image(np.uint8, foreground_value=1))
    other = feret.all(_square_image(np.uint8, foreground_value=foreground_value))
    np.testing.assert_allclose(other, ref, rtol=0, atol=0)


def test_multiple_regions_use_the_convex_hull_of_all_regions() -> None:
    """Documented behaviour: multiple disconnected components are enclosed
    by a single convex hull. maxf must span across them."""

    img = np.zeros((20, 20), dtype=np.uint8)
    img[2:5, 2:5] = 1
    img[15:18, 15:18] = 2  # a second region with a different label value

    maxf = feret.max(img)
    assert maxf == pytest.approx(np.hypot(17 - 2, 17 - 2)), (
        "maxf should be the diagonal spanning all connected components"
    )


def test_empty_mask_raises() -> None:
    """No foreground pixels is not a valid input; must not silently return
    zero or NaN."""

    empty = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(Exception):  # noqa: B017 - characterizing "must fail loudly", any type OK
        feret.calc(empty)


def test_single_pixel_raises() -> None:
    """A single-pixel foreground has too few points for the current convex
    hull based algorithm. Must fail loudly rather than return garbage."""

    img = np.zeros((10, 10), dtype=np.uint8)
    img[5, 5] = 1
    with pytest.raises(Exception):  # noqa: B017 - characterizing "must fail loudly", any type OK
        feret.calc(img)


def test_import_surface_contains_documented_api() -> None:
    for name in ("calc", "all", "plot", "max", "min", "min90", "max90"):
        assert hasattr(feret, name), f"feret should expose {name!r}"
        assert callable(getattr(feret, name))


def test_input_image_is_not_mutated() -> None:
    """A read-only input must be safe to pass in — the current implementation
    should not write back to the caller's array."""

    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:15, 5:15] = 1
    snapshot = img.copy()
    feret.calc(img)
    np.testing.assert_array_equal(img, snapshot)
