import feret
import numpy as np
import matplotlib.pyplot as plt
import time

img = np.load('img.npy')



######
# Testing without edge
edge = False

# Test calc-method
t0 = time.time()
f = feret.calc(img, edge)
print(f'calc-method succesful. {time.time() - t0} secs')

# Test all-method
t0 = time.time()
maxf, minf = feret.all(img, edge)
print(f'all-method succesful. {time.time() - t0} secs')

# Test max-method
t0 = time.time()
maxf = feret.max(img, edge)
print(f'max-method succesful. {time.time() - t0} secs')

# Test min-method
t0 = time.time()
minf = feret.min(img, edge)
print(f'min-method succesful. {time.time() - t0} secs')

print(f'\nMaxFeret: {maxf}\nMinFeret: {minf}')


######
# Testing with edge
edge = True

# Test calc-method
t0 = time.time()
f = feret.calc(img, edge)
print(f'calc-method succesful. {time.time() - t0} secs')

# Test all-method
t0 = time.time()
maxf, minf = feret.all(img, edge)
print(f'all-method succesful. {time.time() - t0} secs')

# Test max-method
t0 = time.time()
maxf = feret.max(img, edge)
print(f'max-method succesful. {time.time() - t0} secs')

# Test min-method
t0 = time.time()
minf = feret.min(img, edge)
print(f'min-method succesful. {time.time() - t0} secs')

print(f'\nMaxFeret: {maxf}\nMinFeret: {minf}')