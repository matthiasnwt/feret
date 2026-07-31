"""Synthetic binary shapes used by the characterization suite.

Every generator returns a 2-D ``uint8`` array with foreground value ``1`` unless
noted otherwise. Shapes are deliberately small so the characterization suite
runs quickly, but large enough that the convex hull has multiple non-trivial
vertices.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np


def _canvas(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


def square() -> np.ndarray:
    img = _canvas((40, 40))
    img[10:30, 10:30] = 1
    return img


def rectangle() -> np.ndarray:
    img = _canvas((40, 60))
    img[10:30, 10:50] = 1
    return img


def rotated_square() -> np.ndarray:
    """Axis-rotated square: side 20, centered, rotated 30 degrees."""

    canvas_shape = (60, 60)
    cy, cx = canvas_shape[0] / 2, canvas_shape[1] / 2
    half = 10.0
    theta = np.deg2rad(30.0)
    y, x = np.mgrid[0 : canvas_shape[0], 0 : canvas_shape[1]]
    dy = y - cy
    dx = x - cx
    ry = dy * np.cos(theta) + dx * np.sin(theta)
    rx = -dy * np.sin(theta) + dx * np.cos(theta)
    return ((np.abs(rx) <= half) & (np.abs(ry) <= half)).astype(np.uint8)


def thin_diagonal_line() -> np.ndarray:
    img = _canvas((30, 30))
    for i in range(3, 27):
        img[i, i] = 1
    return img


def circle() -> np.ndarray:
    canvas_shape = (60, 60)
    cy, cx = 30, 30
    r = 20
    y, x = np.mgrid[0 : canvas_shape[0], 0 : canvas_shape[1]]
    return (((y - cy) ** 2 + (x - cx) ** 2) <= r * r).astype(np.uint8)


def ellipse() -> np.ndarray:
    canvas_shape = (60, 80)
    cy, cx = 30, 40
    ry, rx = 15, 30
    y, x = np.mgrid[0 : canvas_shape[0], 0 : canvas_shape[1]]
    return ((((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2) <= 1.0).astype(np.uint8)


def l_shape() -> np.ndarray:
    img = _canvas((40, 40))
    img[5:35, 5:15] = 1
    img[25:35, 5:35] = 1
    return img


def concave_c_shape() -> np.ndarray:
    img = _canvas((40, 40))
    img[5:35, 5:35] = 1
    img[10:30, 15:35] = 0
    return img


def cross_shape() -> np.ndarray:
    img = _canvas((40, 40))
    img[15:25, 5:35] = 1
    img[5:35, 15:25] = 1
    return img


def touching_border() -> np.ndarray:
    img = _canvas((30, 30))
    img[0:15, 0:20] = 1
    return img


def random_blob_a() -> np.ndarray:
    """Union of small circles placed at fixed seeded positions."""

    rng = np.random.default_rng(seed=42)
    canvas_shape = (60, 60)
    y, x = np.mgrid[0 : canvas_shape[0], 0 : canvas_shape[1]]
    img = _canvas(canvas_shape)
    for _ in range(8):
        cy = rng.integers(15, 45)
        cx = rng.integers(15, 45)
        r = rng.integers(4, 8)
        img |= (((y - cy) ** 2 + (x - cx) ** 2) <= r * r).astype(np.uint8)
    return img


def random_blob_b() -> np.ndarray:
    rng = np.random.default_rng(seed=7)
    canvas_shape = (80, 80)
    y, x = np.mgrid[0 : canvas_shape[0], 0 : canvas_shape[1]]
    img = _canvas(canvas_shape)
    for _ in range(12):
        cy = rng.integers(10, 70)
        cx = rng.integers(10, 70)
        r = rng.integers(3, 10)
        img |= (((y - cy) ** 2 + (x - cx) ** 2) <= r * r).astype(np.uint8)
    return img


def random_blob_c() -> np.ndarray:
    rng = np.random.default_rng(seed=1234)
    canvas_shape = (50, 70)
    y, x = np.mgrid[0 : canvas_shape[0], 0 : canvas_shape[1]]
    img = _canvas(canvas_shape)
    for _ in range(6):
        cy = rng.integers(10, 40)
        cx = rng.integers(10, 60)
        r = rng.integers(5, 12)
        img |= (((y - cy) ** 2 + (x - cx) ** 2) <= r * r).astype(np.uint8)
    return img


def repo_fixture_image() -> np.ndarray:
    """Load the tests/img.npy fixture that lives in the repository."""

    from pathlib import Path

    path = Path(__file__).with_name("img.npy")
    return np.load(path)


SHAPES: Dict[str, Callable[[], np.ndarray]] = {
    "square": square,
    "rectangle": rectangle,
    "rotated_square": rotated_square,
    "thin_diagonal_line": thin_diagonal_line,
    "circle": circle,
    "ellipse": ellipse,
    "l_shape": l_shape,
    "concave_c_shape": concave_c_shape,
    "cross_shape": cross_shape,
    "touching_border": touching_border,
    "random_blob_a": random_blob_a,
    "random_blob_b": random_blob_b,
    "random_blob_c": random_blob_c,
    "repo_fixture": repo_fixture_image,
}


def all_shapes() -> Dict[str, np.ndarray]:
    return {name: fn() for name, fn in SHAPES.items()}
