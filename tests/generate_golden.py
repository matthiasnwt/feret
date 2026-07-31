"""Regenerate ``tests/golden.json`` from whatever version of ``feret`` is on
the import path.

This is the golden-master oracle for the characterization suite. Every value it
records is treated by the tests as the specification of current behaviour and
must survive the refactor bit-identically.

Run with::

    python tests/generate_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import feret

from shapes import SHAPES

GOLDEN_PATH = Path(__file__).with_name("golden.json")

# Public attributes on the object returned by ``feret.calc``.
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


def _record_one(img, edge: bool) -> Dict[str, Any]:
    res = feret.calc(img, edge=edge)
    record: Dict[str, Any] = {attr: float(getattr(res, attr)) for attr in PUBLIC_ATTRS}

    all_tuple = feret.all(img, edge=edge)
    record["all_tuple"] = [float(v) for v in all_tuple]
    record["max"] = float(feret.max(img, edge=edge))
    record["min"] = float(feret.min(img, edge=edge))
    record["min90"] = float(feret.min90(img, edge=edge))
    record["max90"] = float(feret.max90(img, edge=edge))
    return record


def main() -> None:
    golden: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name, factory in SHAPES.items():
        img = factory()
        golden[name] = {
            "edge_false": _record_one(img, edge=False),
            "edge_true": _record_one(img, edge=True),
        }
        print(f"captured: {name}")

    GOLDEN_PATH.write_text(json.dumps(golden, indent=2, sort_keys=True))
    print(f"\nwrote {GOLDEN_PATH} ({len(golden)} shapes x 2 edge modes)")


if __name__ == "__main__":
    main()
