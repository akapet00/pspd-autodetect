import os

import numpy as np
import open3d as o3d
import pyvista as pv
pv.set_plot_theme('document')


# data
fname = os.path.join('input', 'data', 'head.scaled')
mesh = o3d.io.read_triangle_mesh(fname + '.iso.watertight.off')
mesh = mesh.compute_vertex_normals()
vert = np.asarray(mesh.vertices)
tri = np.asarray(mesh.triangles)
normals = np.asarray(mesh.vertex_normals)
colors = np.round((0.5 * normals + 0.5) * 255.)

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
                     image_scale=3,
                     lighting='none',
                     polygon_smoothing=True)
plotter.camera_position = [
    (42.06692280427686, 50.371956888280145, 22.506975710533908),
    (0.2649998664855957, -0.7945003509521484, -1.9323501586914062),
    (-0.2258435035120387, -0.2634546022385864, 0.9378626682413136)
]
_ = plotter.add_mesh(mesh,
                     scalars=colors)
plotter.add_light(light)
plotter.remove_scalar_bar()

formats = ['pdf', 'png']
for ext in formats:
    fname = os.path.join('figures', f'normals_rgb.{ext}')
    if ext == 'png':
        plotter.screenshot(fname, window_size=[500, 500])
    else:
        plotter.save_graphic(fname)
