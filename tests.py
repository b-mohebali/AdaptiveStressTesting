# #! /usr/bin/python3

from sobol_stuff import * 
import matplotlib.pyplot as plt
import numpy as np 
from samply.hypercube import cvt
n = 200
seed = 13

points = []
for _ in range(n):
    newpoint, seed = i4_sobol(2, seed)
    points.append(newpoint)
points = np.array(points)

samples = cvt(count = n, dimensionality=2, verbose=False)


plt.figure()
# plt.scatter(points[:,0], points[:,1], s = 8)
plt.scatter(samples[:,0], samples[:,1], s = 8, color = 'r')
plt.grid(True)
plt.show()
