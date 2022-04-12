# *Feret*: A Python Module to calculate the Feret Diameter of Binary Images

![Downloads](https://pepy.tech/badge/feret)

This python module can calculate the following parameters for binary images:

* maximum Feret diameter (maxferet, maxf)
* minimum Feret diameter (minferet, minf)
* Feret diameter 90 ° to the minferet (minferet90, minf90) 
* Feret diameter 90 ° to maxferet (maxferet90, maxf90) 

See this [Wikipedia page](https://en.wikipedia.org/wiki/Feret_diameter) to get the definition of those parameters.

This module gives the exact results as ImageJ (use `edge=True` as shown below), all the parameters are exactly calculated and **not** approximated.

## Installations
This project is available via pip:

`pip install feret`

## Pieces of Information

#### Convex Hull

The definition of the maxferet and minferet uses the image of a caliper. Therefore, only the points which correspond to the convex hull of the object play a role. That is why before any calculations the convex hull is determined to reduce the runtime.

#### Maxferet
The maxferet is calculated as the maximum Euclidean distance of all pixels.

#### Minferet

The minferet is exactly calculated and **not** approximated. My algorithm uses the fact, that the calipers that define the minferet run on one side through two points and on the other through one point. The script iterates over all edge points and defines a line through the point and the one next to it. Then all the distances to the other points are calculated and the maximum is taken. The minimum of all those maximums is the minferet. The maximum of all those maximums is **not** the maxferet, that is the reason it is calculated separately. The runtime of this is already pretty good but hopefully I can improve it in the future.

## Use
The module can be used as followed:

First you need a binary image for which the feret diameter should be calculated. The background has to have the value zero, the object can have any nonzero  value. The object doesn't have to be convex. At the moment the module only supports one object per image.This means, that if there are multiple not connected regions, the script will calculate a convexhull which include all regions and for this hull the feret diameter is calculated.

Thr calls are:

```python
import feret

# tifffile is not required nor included in this module.
import tifffile as tif
img = tif.imread('example.tif') # Image has to be a numpy 2d-array.


# get the values
maxf, minf, minf90, maxf90 = feret.all(img)

# get only maxferet
maxf = feret.max(img)

# get only minferet
minf = feret.min(img)

# get only minferet90
minf90 = feret.min90(img)

# get only maxferet90
maxf90 = feret.max90(img)

# get all the informations
res = feret.calc(img)
maxf = res.maxf
minf =  res.minf
minf90 = res.minf90
minf_angle = res.minf_angle
minf90_angle = res.minf90_angle
maxf_angle = res.maxf_angle
maxf90_angle = res.maxf90_angle
```

There is an option to calculate the Feret diameters for the pixel edges instead of the centers. Just add an `edge=True` in the call as shown below. This works for all calls analogous.

```python
import feret

# tifffile is not required nor included in this module.
import tifffile as tif
img = tif.imread('example.tif') # Image has to be a numpy 2d-array.

# get only maxferet
maxf = feret.max(img, edge=True)
```

### Future Releases

* add a return_angle in the calls
* add the possibility to show plots of the results
