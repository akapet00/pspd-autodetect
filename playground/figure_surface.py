import os

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy as np
import open3d as o3d


elev, azim = 25, 50
bounding_box = False

# data
fname = os.path.join('input', 'data', 'head.scaled')
points = np.loadtxt(fname + '.xyz')
normals = np.loadtxt(fname + '.normals')
mesh = o3d.io.read_triangle_mesh(fname + '.iso.watertight.off')

# bounding box
xmin, ymin, zmin = np.min(points, axis=0)
xmax, ymax, zmax = np.max(points, axis=0)
A = [xmax, ymin, zmin]
B = [xmax, ymax, zmin]
C = [xmin, ymax, zmin]
D = [xmin, ymin, zmin]
E = [xmax, ymin, zmax]
F = [xmax, ymax, zmax]
G = [xmin, ymax, zmax]
H = [xmin, ymin, zmax]

# surface
mesh_smp = mesh.simplify_quadric_decimation(6500)
vert_smp = np.asarray(mesh_smp.vertices)
tri_smp = np.asarray(mesh_smp.triangles)
ls = LightSource(elev+20, azim)

# surface in a bounding box
fig = plt.figure(figsize=(5, 5))
ax = plt.axes(projection='3d')
ax.plot_trisurf(*vert_smp.T, triangles=tri_smp,
                color='gray', ec='k', lw=0.05, lightsource=ls)
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
    fig.savefig(os.path.join('figures', f'surface.{ext}'),
                dpi=300,
                bbox_inches='tight',
                pad_inches=None)
