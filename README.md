# *Feret*: A Python Module to calculate the Feret Diameter of Binary Images

This python module can calculate the maximum Feret diameter (maxferet) and minimum Feret diameter (minferet) of a binary image. For a detailed explanation see this [wikipedia page](https://en.wikipedia.org/wiki/Feret_diameter).

## Installation
This project is available via pip

`pip install feret`

## Informations

#### Maxferet
The maxferet is calculated as the maximum Euclidean distance of all pixels.

#### Minferet
The minferet is only approximated in two steps at the moment. First, the distance of to parallel lines, which surround the object, are calculated for all angles from 0° to 180°. The minimum of this first calculation is used as the initial guess for a minimization algorithm, which is the second part of the approximation. Even if this method is not perfect, the difference to the true minferet can be neglected for most cases.


At this early development stage, it can only calculate the maximum and minimum Feret Diameter but feature releases will offer the Feret diameter 90° to maximum and minimum. The module will also not return the angle of the diameters. Many things will come in the future.

## Use
The module can be used as followed:

```python
import feret

# tifffile is not required nor included in this module.
import tifffile as tif
img = tif.imread('example.tif') # Image has to be a numpy 2d-array.

# get the class
res = feret.calc(img)
maxf, minf = res.maxferet, res.minferet

# get the values
maxf, minf = feret.all(img)

# get only maxferet
maxf = feret.max(img)

# get only minferet
minf = feret.min(img)
```

At the moment there is only one option. It is possible to use the pixel corners instead of the pixel centers. ImageJ uses pixel corners. Here the keyword `edge` is used. See the following code to get maxferet and minferet for the edges.

```python
import feret

# tifffile is not required nor included in this module.
import tifffile as tif
img = tif.imread('example.tif') # Image has to be a numpy 2d-array.

# get the class
res = feret.calc(img, edge=True)
maxf, minf = res.maxferet, res.minferet

# get the values
maxf, minf = feret.all(img, edge=True)

# get only maxferet
maxf = feret.max(img, edge=True)

# get only minferet
minf = feret.min(img, edge=True)
```

