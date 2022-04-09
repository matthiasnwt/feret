# Feret

This python module can calculate the maximum and minimum Feret Diameter of a binary image. For a detailed explanation see this [wikipedia page](https://en.wikipedia.org/wiki/Feret_diameter).

At this early development stage, it can only calculate the maximum and minimum Feret Diameter but feature releases will offer the Feret diameter 90° to maximum and minimum. The module will also not return the angle of the diameters. Many things will come in the future.

The module can be used as followed:

```python
import feret
import tifffile as tif

img = tif.imread('example.tif')

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

At the moment there is only one option. It is possible to use the pixel corners instead of the pixel centers. ImageJ uses the pixel corners. Here the keyword `edge` is used. See the following code to get maxferet und minferet for the edges.

```python
import feret
import tifffile as tif

img = tif.imread('example.tif')

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

