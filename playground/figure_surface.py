import os

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy as np
import open3d as o3d


elev, azim = 25, 50

# data
fname = os.path.join('input', 'data', 'head.scaled')
points = np.loadtxt(fname + '.xyz')
normals = np.loadtxt(fname + '.normals')
mesh = o3d.io.read_triangle_mesh(fname + '.iso.watertight.off')

# surface
mesh_smp = mesh.simplify_quadric_decimation(6500)
vert_smp = np.asarray(mesh_smp.vertices)
tri_smp = np.asarray(mesh_smp.triangles)
ls = LightSource(elev+20, azim)

fig = plt.figure(figsize=(5, 5))
ax = plt.axes(projection='3d')
ax.plot_trisurf(*vert_smp.T, triangles=tri_smp,
                color='gray', ec='k', lw=0.05, lightsource=ls)
ax.set_box_aspect(np.ptp(points, axis=0))
ax.set_axis_off()
ax.view_init(elev, azim)
plt.show()

formats = ['png', 'pdf']
for ext in formats:
    fig.savefig(os.path.join('figures', f'surface.{ext}'),
                dpi=350,
                bbox_inches='tight',
                pad_inches=None)
