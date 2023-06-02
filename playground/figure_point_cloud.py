import os

import matplotlib.pyplot as plt
import numpy as np


elev, azim = 25, 50
bounding_box = True

# data
fname = os.path.join('input', 'data', 'head.scaled')
points = np.loadtxt(fname + '.xyz')
N = points.shape[0]
num = 0.1 * N
mask = np.arange(0, N, int(N/num))
x, y, z = points[mask].T

# bounding box
xmin, ymin, zmin = np.min(points[mask], axis=0)
xmax, ymax, zmax = np.max(points[mask], axis=0)
A = [xmax, ymin, zmin]
B = [xmax, ymax, zmin]
C = [xmin, ymax, zmin]
D = [xmin, ymin, zmin]
E = [xmax, ymin, zmax]
F = [xmax, ymax, zmax]
G = [xmin, ymax, zmax]
H = [xmin, ymin, zmax]

# point cloud in a bounding box
fig = plt.figure(figsize=(5, 5))
ax = plt.axes(projection='3d')
ax.scatter(x, y, z,
           fc='w', ec='k', s=5, lw=0.5, rasterized=True)
if bounding_box:
    ax.plot(*np.c_[A, B], c='k', lw=1)
    ax.plot(*np.c_[A, D], c='k', lw=1)
    ax.plot(*np.c_[A, E], c='k', lw=1)
    ax.plot(*np.c_[B, C], c='k', lw=1)
    ax.plot(*np.c_[B, F], c='k', lw=1, zorder=3)
    ax.plot(*np.c_[C, D], c='k', lw=1)
    ax.plot(*np.c_[C, G], c='k', lw=1)
    ax.plot(*np.c_[D, H], c='k', lw=1)
    ax.plot(*np.c_[E, F], c='k', lw=1, zorder=3)
    ax.plot(*np.c_[E, H], c='k', lw=1)
    ax.plot(*np.c_[F, G], c='k', lw=1, zorder=3)
    ax.plot(*np.c_[G, H], c='k', lw=1)
ax.set_box_aspect(np.ptp(points, axis=0))
ax.set_axis_off()
ax.view_init(elev, azim)
plt.show()


formats = ['png', 'pdf']
for ext in formats:
    fig.savefig(os.path.join('figures', f'point_cloud.{ext}'),
                dpi=300,
                bbox_inches='tight',
                pad_inches=None)
