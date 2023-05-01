import argparse
import os

import numpy as np
try:
    import open3d as o3d
except ImportError:
    raise ImportError('`open3d` not installed')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-knn',
                        type=int,
                        default=30,
                        help='number of nearest neighbors to be searched')
    parser.add_argument('-fnc',
                        '--fast_normal_computation',
                        action='store_true',
                        help='enable the non-iterative normal estimation')
    parser.add_argument('-s',
                        '--save',
                        action='store_true',
                        help='save estimated normals')
    args = parser.parse_args()
    return args
    

def main():
    args = parse_args()
    knn = args.knn
    fnc = args.fast_normal_computation
    save = args.save
    fname = os.path.join('model', 'head')
    points = np.loadtxt(fname + '.xyz', delimiter=',')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=knn),
                         fast_normal_computation=fnc)
    pcd = pcd.normalize_normals()
    pcd.orient_normals_consistent_tangent_plane(k=knn)
    normals = np.asarray(pcd.normals)
    if save:
        np.savetxt(fname + '.normals', normals, delimiter=',')
        print(f'Estimated normals saved to `{fname}.normals`.')


if __name__ == '__main__':
    main()
