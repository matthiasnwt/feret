# FINDINGS

This file lists things noticed during the 1.3.1 → 1.4.0 refactor that look
like genuine bugs or design smells in the pre-refactor code. **None of them
have been fixed** — the refactor preserves the observable behaviour of every
one of them bit-identically, per the refactor's non-negotiable constraint
that no user of the library should have to change any code.

Each item states what the current behaviour is, why it looks wrong, and what
a hypothetical follow-up fix would need to do. Fixes should be discussed and
released as a proper minor or major version bump, not smuggled into a
refactor.

---

## 1. `ndimage.center_of_mass(self.hull)` is mathematically meaningless

**Where:** the original `feret/main.py`, now preserved as
`_hull.hull_pseudo_center` and consumed as `FeretResult.y0` / `FeretResult.x0`.

**What it does today:**

```python
self.y0, self.x0 = ndimage.center_of_mass(self.hull)
```

`self.hull` is a `(2, N)` NumPy array of hull point coordinates, not a
weighted image. Feeding it to `ndimage.center_of_mass` treats it as a mass
distribution over its two-element index axis, producing values that are a
function of `sum(hull_ys)` and `sum(hull_xs)` in a way that has nothing to do
with the geometric center of the hull.

**Why it does not corrupt any output:** the values enter
`_diameters.caliper_extents_at_angle` (formerly `Calculater.calculate_distances`)
only as a constant offset added to `ds`, and `argmax`/`argmin` cancel any
constant offset:

```python
ds = np.cos(a) * (self.y0 - self.hull[0]) - np.sin(a) * (self.x0 - self.hull[1])
      = const + (-cos(a) * self.hull[0] + sin(a) * self.hull[1])
```

So the perpendicular Feret computations do not actually depend on `y0` /
`x0`. The value is preserved verbatim because it is *observable* — an
attribute on the returned object — and removing it could break isinstance-
style introspection that a user might have written. A follow-up cleanup could
either (a) delete `y0` / `x0` entirely, (b) replace them with the correct
centroid of the hull, or (c) rename them to something like `_unused_offset`
to document that they carry no useful information.

## 2. `feret.all` docstring says two return values, code returns four

**Where:** the pre-refactor `feret/__init__.py`::

```python
def all(img, edge=False):
    """
    ...
    Returns:
        maxf (float): maximum feret diameter
        minf (float): minimum feret diameter
    """
    feret_calc = calc(img, edge)
    return feret_calc.maxf, feret_calc.minf, feret_calc.minf90, feret_calc.maxf90
```

The docstring documents two return values, but the code returns four. This
is why the old `tests/tests.py` did `maxf, minf = feret.all(...)` — that
line would have raised `ValueError: too many values to unpack` on any real
input.

**What was done:** the refactored docstring says "Return
`(maxf, minf, minf90, maxf90)` in that order." The behaviour is unchanged;
only the documentation is corrected. Documentation-only fixes are not
observable behaviour changes.

## 3. Missing `.T` in the special-case branch of `minf90_t`

**Where:** the pre-refactor `calculate_minferet90`::

```python
if self.minf90_angle == 0 or self.minf90_angle == np.pi:
    self.minf90_t = self.minf90_coords[0][0]        # note: no .T
```

The sibling method `calculate_maxferet90` writes the equivalent line as::

```python
    self.maxf90_t = self.maxf90_coords.T[0][0]      # note: has .T
```

For a `(2, 2)` coord array both indexers happen to return the same element
(`coords[0, 0]`), so the numerical result is identical and the tests
correctly assert on it. It is preserved verbatim in
`_diameters.perpendicular_feret`, which uses `coords.T[0][0]` for both cases.
This is a cosmetic inconsistency, not a bug that changes results.

## 4. `feret.calc` on an empty mask raises a cryptic error

**Where:** convex-hull extraction returns `None` from `cv2.convexHull` when
there are no foreground points, and the code then does `.T` on it and blows
up with `AttributeError: 'NoneType' object has no attribute 'T'`.

**What was done:** preserved. The characterization suite (see
`tests/test_edge_cases.py::test_empty_mask_raises`) asserts only that
*some* exception is raised — the exact type is not part of the API contract.
A follow-up fix should raise a clear `ValueError("input mask has no
foreground pixels")` at the top of `calc`.

## 5. `feret.calc` on a single-pixel mask raises an equally cryptic error

**Where:** a single foreground pixel produces a convex hull of one point,
after which `hull.T[(i + 1) % length]` cannot find a second edge point and
the algorithm fails inside `min_feret`. Preserved bit-for-bit; same
follow-up recommendation as (4).

## 6. `matplotlib` was imported at package top level

**Where:** the pre-refactor `feret/__init__.py` had `from matplotlib import
pyplot as plt` at the top, even though it was not used inside `__init__.py`.
That meant `import feret` always paid the matplotlib import cost.

**What was done:** matplotlib is now only imported inside `feret.plot`,
matching the refactor brief ("keep plotting isolated so matplotlib is only
imported where it is actually needed"). This is a startup-cost improvement
with no observable change to any Feret value. Matplotlib is still a hard
runtime dependency, so `pip install feret; python -c "import feret;
feret.plot(img)"` still works exactly as before.

## 7. NumPy 2.0 removed `np.cross` on 2-D vectors

**Where:** `Calculater.calculate_minferet` called `np.cross(p2 - p1,
p1 - self.hull.T)` where both inputs are 2-D. NumPy 2.0 removed that
overload; the code raised `ValueError` on any input under NumPy 2.x.

**What was done:** replaced with the inline scalar formula
`v[0]*w[:, 1] - v[1]*w[:, 0]`, which is what `np.cross` used to compute
internally. Because it is the same operation on the same operands in the
same order, the result is bit-identical on NumPy 1.x and the package now
also runs under NumPy 2.x. The characterization suite passes on both
NumPy 1.26.4 and NumPy 2.5.1, and the A/B script (`scripts/ab_compare.py`)
confirms bit-identical outputs across 500 shapes against the 1.3.1 baseline.

This is the *only* line of numerical code the refactor changed, and it was
changed because the original code did not run at all under a currently
supported dependency version.
