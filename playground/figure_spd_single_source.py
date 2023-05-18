import os

import numpy as np
import open3d as o3d
from pspd import PSPD
import pyvista as pv
pv.set_plot_theme('document')
import seaborn as sns

from single_source import generate_power_density
from experiment_single_source import (AMPLITUDE,
                                      RADIUS,
                                      QUERY_POINT,
                                      PROJECTED_AREA,
                                      SCALER)


cmap = sns.color_palette(palette='viridis', as_cmap=True)

# data
fname = os.path.join('input', 'data', 'head.scaled')
mesh = o3d.io.read_triangle_mesh(fname + '.iso.watertight.off')
vert = np.asarray(mesh.vertices)
tri = np.asarray(mesh.triangles)

# generate power density
power_density = generate_power_density(AMPLITUDE,
                                       RADIUS,
                                       QUERY_POINT,
                                       vert,
                                       SCALER)

# find pspd using vertices of a mesh
pspd = PSPD(vert, power_density, mesh=mesh)
pspd.find(PROJECTED_AREA)
ind, _ = pspd.get_points()
res = pspd.get_results(peak=False)
colors = np.asarray(res['spatially averaged power density'])
colors[colors < 0] = 0
colors = colors / colors.max()

# surface
mesh = pv.PolyData()
mesh.points = vert
mesh.faces = np.append(np.full((tri.shape[0], 1), 3, dtype=np.int32),
                       tri,
                       axis=1)
# light
light = pv.Light(light_type='headlight')
# plotter settings
plotter = pv.Plotter(notebook=False,
                     off_screen=True,
                     image_scale=6,
                     lighting='none',
                     polygon_smoothing=True)
plotter.camera_position = [
    (59.627623746536464, 32.91796546447344, 15.45259422927298),
    (0.2649998664855957, -0.7945003509521484, -1.9323501586914062),
    (-0.20614719426313596, -0.13666631768542525, 0.9689301584261598)
]
sargs = dict(
    position_x=0.175,
    position_y=0.05,
    height=0.1,
    width=0.65,
    n_labels=0,
)
_ = plotter.add_mesh(mesh,
                     scalars=colors,
                     cmap=cmap,
                     scalar_bar_args=sargs)
plotter.add_light(light)

formats = ['pdf', 'png']
for ext in formats:
    fname = os.path.join('figures', f'spd_single_source.{ext}')
    if ext == 'png':
        plotter.screenshot(fname, window_size=[500, 500])
    else:
        plotter.save_graphic(fname)
