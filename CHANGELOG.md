# Changelog

## 1.4.0

**Pure refactor — no behavioural change.**

The internals were restructured for maintainability, but every value returned
by the public API is bit-identical to what version 1.3.1 returned. This was
verified in two ways:

* a 291-test characterization suite compares the current output to golden
  values captured from 1.3.1 (`rtol=0`, `atol=0`),
* the `scripts/ab_compare.py` script generates 500 random binary shapes and
  compares 16 fields per shape against a baseline recorded from 1.3.1;
  every value matches bit-identically.

### Structural changes

* Adopted a `src/` layout under `src/feret/`.
* Split the monolithic `feret/main.py::Calculater` into focused modules:
  * `feret._hull` — convex-hull extraction (center vs. edge convention),
  * `feret._diameters` — Feret-diameter algorithms,
  * `feret._result` — `FeretResult` (return type of `feret.calc`),
  * `feret._plotting` — matplotlib rendering (imported lazily),
  * `feret._api` — public entry points wired up on top of the above.
* `feret.__init__` is now a thin façade that re-exports the seven public
  entry points and `__version__`.
* Type hints throughout, consistent NumPy-style docstrings.
* Matplotlib is no longer imported when you `import feret` — only inside
  `feret.plot(...)`.

### Compatibility

* `from feret.main import Calculater` still works (shim in
  `src/feret/main.py`).
* All seven public entry points keep their names, argument order, keyword
  names, defaults, return types and return order:
  `feret.calc`, `feret.all`, `feret.plot`, `feret.max`, `feret.min`,
  `feret.min90`, `feret.max90`.
* The one internal code change is the removal of a call to `np.cross` on
  2-D vectors (removed from NumPy 2.0); it was replaced with the identical
  scalar formula. See `FINDINGS.md` §7.

### Tooling

* `pyproject.toml` migrated from Poetry to PEP 621 (setuptools backend).
* `ruff` and `mypy` configured; `pre-commit` hooks added.
* GitHub Actions CI runs the test suite on Python 3.9, 3.10, 3.11 and 3.12
  across Linux, macOS and Windows, plus a lint job.
* Requires Python 3.9+.

### Known behaviour preserved verbatim

Six pre-existing quirks are documented in `FINDINGS.md` and were
deliberately left untouched. See that file if you plan a follow-up release
that fixes them.

## 0.3.3

minferet90 added
