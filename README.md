# Feret

This python module can calculate the maximum and minimum Feret Diameter of a binary image. For a detailed explanation see this [wikipedia page](https://en.wikipedia.org/wiki/Feret_diameter).

At this early development stage, it can only calculate the maximum and minimum Feret Diameter but feature releases will offer the Feret diameter 90° to maximum and minimum. The module will also not return the angle of the diameters. Many things will come in the future.

The module can be used as followed:

```python
import feret
import tifffile as tif

img = tif.imread('example.tif')

res = feret.calc(img)

print(res.maxferet, res.minferet)
```

