import os

import numpy as np
import open3d as o3d
import pickle
from pspd import PSPD

from single_source import generate_power_density


# constants
AMPLITUDE = 10  # W/m2
RADIUS = 2.5   # cm
QUERY_POINT = np.array([8.4082, -3.0716, -1.8224])  # cm
PROJECTED_AREA = 4  # cm2
SCALER = [1, 0.5, 0.25]


def main():
    # data
    fname = os.path.join('input', 'data', 'head.scaled')
    points = np.loadtxt(fname + '.xyz') 
    normals = np.loadtxt(fname + '.normals')
    mesh = o3d.io.read_triangle_mesh(fname + '.iso.watertight.off')
    vert = np.asarray(mesh.vertices)
    tri = np.asarray(mesh.triangles)

    # generate power density
    power_density = generate_power_density(AMPLITUDE,
                                           RADIUS,
                                           QUERY_POINT,
                                           points,
                                           SCALER)
    colors = generate_power_density(AMPLITUDE,
                                    RADIUS,
                                    QUERY_POINT,
                                    vert,
                                    SCALER)

    # find pspd
    pov = np.mean(points, axis=0)
    diameter = np.linalg.norm(points.ptp(axis=0))
    pov[0] += 2 * diameter
    pov[1] += 0.5 * diameter

    pspd = PSPD(points, power_density, mesh=mesh)
    pspd.find(PROJECTED_AREA, pov=pov, p=np.pi)
    ind, _ = pspd.get_points()
    res = pspd.get_results()
    p = res['query point']
    nbh = res['k-neighbourhood']
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
        os.path.join('output', 'experiment_single_source.pkl'), 'wb'
    ) as handle:
        pickle.dump(datadict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
