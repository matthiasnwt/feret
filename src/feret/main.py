"""Backward-compatibility shim.

Legacy code may import ``from feret.main import Calculater``. After the
refactor the result type lives in :mod:`feret._result` under a new name;
this module re-exports it so no downstream import needs to change.
"""

from __future__ import annotations

from ._result import Calculater, FeretResult

__all__ = ["Calculater", "FeretResult"]
