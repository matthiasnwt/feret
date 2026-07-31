"""A/B comparison against the previous release of ``feret``.

This script exists to prove that the refactor is behaviour-preserving on a
much larger batch of inputs than the characterization test suite covers. It
generates a deterministic corpus of 500 binary shapes and runs the current
:mod:`feret` over them, either recording the outputs to a JSON file or
comparing them against a pre-recorded baseline.

Typical A/B workflow (from a clean checkout of the repo)::

    # 1. install the old release into a scratch venv and record its outputs
    python -m venv .venv-old
    .venv-old\\Scripts\\pip install "feret==1.3.1" "numpy<2"
    .venv-old\\Scripts\\python scripts/ab_compare.py \\
        --record scripts/ab_baseline_1.3.1.json

    # 2. install the current (refactored) release and compare
    python -m venv .venv-new
    .venv-new\\Scripts\\pip install -e ".[dev]"
    .venv-new\\Scripts\\python scripts/ab_compare.py \\
        --check scripts/ab_baseline_1.3.1.json

A pre-recorded baseline (``scripts/ab_baseline_1.3.1.json``) is committed to
the repository so that step 2 is enough to reproduce the comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import feret

BATCH_SIZE = 500
GLOBAL_SEED = 20260731

# Fields captured for every shape / edge-mode combination.
FIELDS = (
    "maxf",
    "minf",
    "minf90",
    "maxf90",
    "maxf_angle",
    "minf_angle",
    "maxf90_angle",
    "minf90_angle",
)


def _random_blob(rng: np.random.Generator) -> np.ndarray:
    """Generate one seeded random binary blob (union of small discs)."""

    h = int(rng.integers(30, 100))
    w = int(rng.integers(30, 100))
    n = int(rng.integers(3, 15))
    y, x = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w), dtype=np.uint8)
    for _ in range(n):
        cy = int(rng.integers(5, h - 5))
        cx = int(rng.integers(5, w - 5))
        r = int(rng.integers(3, min(15, min(h, w) // 3)))
        img |= (((y - cy) ** 2 + (x - cx) ** 2) <= r * r).astype(np.uint8)
    # Reject degenerate shapes (empty or single-pixel); very rare but possible.
    if np.count_nonzero(img) < 4:
        return _random_blob(rng)
    return img


def _corpus() -> list[np.ndarray]:
    rng = np.random.default_rng(GLOBAL_SEED)
    return [_random_blob(rng) for _ in range(BATCH_SIZE)]


def _record_one(img: np.ndarray, edge: bool) -> dict[str, float]:
    res = feret.calc(img, edge=edge)
    return {name: float(getattr(res, name)) for name in FIELDS}


def _record_all() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for i, img in enumerate(_corpus()):
        records.append(
            {
                "index": i,
                "shape": list(img.shape),
                "nonzero": int(np.count_nonzero(img)),
                "edge_false": _record_one(img, edge=False),
                "edge_true": _record_one(img, edge=True),
            }
        )
    return records


def _compare(new: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> int:
    if len(new) != len(baseline):
        print(f"[FAIL] length mismatch: new={len(new)} baseline={len(baseline)}")
        return 1

    mismatches: list[str] = []
    for a, b in zip(new, baseline):
        if a["shape"] != b["shape"]:
            mismatches.append(f"#{a['index']}: canvas shape drift {a['shape']} != {b['shape']}")
            continue
        for edge_key in ("edge_false", "edge_true"):
            for field in FIELDS:
                va = a[edge_key][field]
                vb = b[edge_key][field]
                if not np.isclose(va, vb, rtol=0.0, atol=0.0, equal_nan=True):
                    mismatches.append(
                        f"#{a['index']} {edge_key} {field}: new={va!r} baseline={vb!r}"
                    )

    if mismatches:
        print(f"[FAIL] {len(mismatches)} mismatches on {len(new)} shapes")
        for m in mismatches[:20]:
            print("  " + m)
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more")
        return 1

    print(f"[OK] all {len(new)} shapes match baseline bit-identically ({len(FIELDS) * 2} fields each)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--record", metavar="PATH", type=Path, help="write feret outputs to PATH")
    group.add_argument(
        "--check", metavar="PATH", type=Path, help="compare feret outputs against baseline at PATH"
    )
    args = parser.parse_args()

    if args.record is not None:
        records = _record_all()
        args.record.write_text(json.dumps(records, indent=2))
        print(f"wrote {len(records)} records to {args.record}")
        return 0

    baseline = json.loads(args.check.read_text())
    new = _record_all()
    return _compare(new, baseline)


if __name__ == "__main__":
    sys.exit(main())
