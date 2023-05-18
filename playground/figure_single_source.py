import os

import numpy as np
import pickle
import pyvista as pv
pv.set_plot_theme('document')
import seaborn as sns


cmap = sns.color_palette(palette='viridis', as_cmap=True)

# data
with open(
    os.path.join('output', 'experiment_single_source.pkl'), 'rb'
) as handle:
    datadict = pickle.load(handle)
vert = datadict['vertices']
tri = datadict['faces']
colors = datadict['colors']

# surface
mesh = pv.PolyData()
mesh.points = vert
mesh.faces = np.append(np.full((tri.shape[0], 1), 3, dtype=np.int32),
                       tri,
                       axis=1)
# light
light = pv.Light(light_type='headlight')
# plotter settings
plotter = pv.Plotter(off_screen=True,
                     image_scale=6,
                     lighting='none',
                     polygon_smoothing=True)
plotter.camera_position = [
    (59.627623746536464, 32.91796546447344, 15.45259422927298),
    (0.2649998664855957, -0.7945003509521484, -1.9323501586914062),
    (-0.20614719426313596, -0.13666631768542525, 0.9689301584261598)
]
_ = plotter.add_mesh(mesh,
                     scalars=colors,
                     cmap=cmap)
plotter.add_light(light)
plotter.remove_scalar_bar()

formats = ['pdf', 'png']
for ext in formats:
    fname = os.path.join('figures', f'single_source.{ext}')
    if ext == 'png':
        plotter.screenshot(fname, window_size=[500, 500])
    else:
        plotter.save_graphic(fname)
