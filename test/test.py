import FeretDiameter
import tifffile as tif
import time

path = r'D:\Arbeit\nPSize\FeretDiameter\10_binary_verrgroessert.tif'

image = tif.imread(path)


# maxferet, minferet = FeretDiameter.calculate(
#     image, precision=1, edge=True)

# maxferet, minferet = FeretDiameter.calculate(
#     image, precision=0.1, edge=True)

# maxferet, minferet = FeretDiameter.calculate(
#     image, precision=0.01, edge=True)

# maxferet, minferet = FeretDiameter.calculate(
#     image, precision=0.001, edge=True)

start = time.perf_counter()
öö
maxferet, minferet = FeretDiameter.calculate(
    image, precision=0.1, edge=True)

print(time.perf_counter() - start)

print(maxferet, minferet)

# x = 1.9, y=130