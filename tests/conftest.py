"""Shared pytest configuration for the characterization suite."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend so plot() does not open a window
