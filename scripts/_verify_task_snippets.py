"""Run the exact snippets from the refactor task prompt against the
refactored package to prove the public API is unchanged.

Not part of the production package — a one-shot verification script kept in
the repo for reproducibility.
"""

from __future__ import annotations

import numpy as np

import feret

# The task fixes an image loaded from a tif; we substitute a synthetic square
# so the script needs no external file. Everything else is copy-pasted from
# the "Public API that must remain unchanged" section of the task prompt.
img = np.zeros((30, 30), dtype=np.uint8)
img[5:25, 5:25] = 1

# get the values
maxf, minf, minf90, maxf90 = feret.all(img)

# get only maxferet
maxf_only = feret.max(img)

# get only minferet
minf_only = feret.min(img)

# get only minferet90
minf90_only = feret.min90(img)

# get only maxferet90
maxf90_only = feret.max90(img)

# get all the informations
res = feret.calc(img)
maxf = res.maxf
minf = res.minf
minf90 = res.minf90
minf_angle = res.minf_angle
minf90_angle = res.minf90_angle
maxf_angle = res.maxf_angle
maxf90_angle = res.maxf90_angle

print("all():", (maxf, minf, minf90, maxf90))
print("max only:", maxf_only, "min only:", minf_only)
print("min90 only:", minf90_only, "max90 only:", maxf90_only)
print("angles:", maxf_angle, minf_angle, maxf90_angle, minf90_angle)

# edge=True keyword
maxf_edge = feret.max(img, edge=True)
print("max(edge=True):", maxf_edge)

# plot() and plot(edge=True) — use Agg so nothing pops up
import matplotlib

matplotlib.use("Agg")
feret.plot(img)
feret.plot(img, edge=True)
print("plot / plot(edge=True) OK")
