import os

import matplotlib.pyplot as plt
import numpy as np


elev, azim = 25, 50

# data
fname = os.path.join('data', 'model', 'head.scaled')
points = np.loadtxt(fname + '.xyz') 

# point cloud
N = points.shape[0]
num = 0.1 * N
mask = np.arange(0, N, int(N/num))

fig = plt.figure(figsize=(5, 5))
ax = plt.axes(projection='3d')
ax.scatter(*points[mask].T,
           fc='w', ec='k', s=7.5, lw=0.5)
ax.set_box_aspect(np.ptp(points, axis=0))
ax.set_axis_off()
ax.view_init(elev, azim)
plt.show()
