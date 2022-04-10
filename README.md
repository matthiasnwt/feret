# *Feret*: A Python Module to calculate the Feret Diameter of Binary Images

This python module can calculate the following parameters for binary images:

* maximum Feret diameter (maxferet, maxf)
* minimum Feret diameter (minferet, minf)
* Feret diameter 90 ° to the minferet (minferet90, minf90) 
* Feret diameter 90 ° to maxferet (maxferet90, maxf90) 

See this [Wikipedia page](https://en.wikipedia.org/wiki/Feret_diameter) to get a the definition of those parameters.

## Installation
This project is available via pip:

`pip install feret`

## Informations

#### Maxferet
The maxferet is calculated as the maximum Euclidean distance of all pixels.

#### Minferet
The minferet is only approximated in two steps at the moment. First, the distance of to parallel lines, which surround the object, is calculated for all angles from 0° to 180°. The minimum of this first calculation is used as the initial guess for a minimization algorithm, which is the second part of the approximation. Even if this method is not perfect, the difference to the true minferet can be neglected in most cases.


At this early development stage, it can only calculate the maximum and minimum Feret Diameter but feature releases will offer the Feret diameter 90° to maximum and minimum. The module will also not return the angle of the diameters. Many things will come in the future.

## Use
The module can be used as followed:

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

