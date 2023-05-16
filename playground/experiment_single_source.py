import os

import numpy as np
import open3d as o3d
import pickle
import seaborn as sns

from pspd import PSPD


# data
fname = os.path.join('input', 'data', 'head.scaled')
points = np.loadtxt(fname + '.xyz') 
normals = np.loadtxt(fname + '.normals')
mesh = o3d.io.read_triangle_mesh(fname + '.iso.watertight.off')
vert = np.asarray(mesh.vertices)
tri = np.asarray(mesh.triangles)


# a single source
def generate_power_density(amplitude,
                           radius,
                           query_point,
                           points,
                           scaler=[1, 1, 1]):
    distance = np.linalg.norm((points - query_point)
                              / np.array(scaler),
                          axis=1)
    return amplitude * np.exp(-(distance / radius) ** 2)


# generate power density
amplitude = 10
radius = 2.5
query_point = np.array([ 8.4082, -3.0716, -1.8224])
power_density = generate_power_density(amplitude,
                                       radius,
                                       query_point,
                                       points,
                                       scaler=[1, 0.5, 0.25])
colors = generate_power_density(amplitude,
                                radius,
                                query_point,
                                vert,
                                scaler=[1, 0.5, 0.25])

# find pspd
pov = np.mean(points, axis=0)
diameter = np.linalg.norm(points.ptp(axis=0))
pov[0] += 2 * diameter
pov[1] += 0.5 * diameter
projected_area = 4
a = np.sqrt(projected_area)

pspd = PSPD(points, power_density, mesh=mesh)
pspd.find(projected_area, pov=pov, p=np.pi)
ind, _ = pspd.get_points()
res = pspd.get_results()
p = res['query point']
nbh = res['k-neigborhood']
area = res['surface area']
spd = res['spatially averaged power density']

# save results
datadict = {'points': points,
            'normals': normals,
            'vertices': vert,
            'colors': colors,
            'faces': tri,
            'power density': power_density,
            'search space indices': ind,
            'query point': p,
            'neighborhood': nbh,
            'surface area': area,
            'pspd': spd}

with open(
    os.path.join('output', __file__.strip('.py') + '.pkl'), 'wb'
) as handle:
    pickle.dump(datadict, handle, protocol=pickle.HIGHEST_PROTOCOL)
