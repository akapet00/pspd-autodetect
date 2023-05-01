import argparse
import os

import numpy as np
try:
    import open3d as o3d
except ImportError:
    raise ImportError('`open3d` not installed')
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth',
                        type=int,
                        default=8,
                        help='maximum depth of the octree')
    parser.add_argument('--scale',
                        type=float,
                        default=1.1,
                        help='input scaler to a uniform size')
    parser.add_argument('--n_threads',
                        type=int,
                        default=1,
                        help='number of threads used during reconstruction')
    parser.add_argument('--write_ascii',
                        action='store_true',
                        help='use ascii format for writing the mesh')
    parser.add_argument('--save',
                        action='store_true',
                        help='save the reconstructed mesh')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    depth = args.depth
    scale = args.scale
    n_threads = args.n_threads
    write_ascii = args.write_ascii
    save = args.save
    fname = os.path.join('model', 'head')
    points = np.loadtxt(fname + '.xyz', delimiter=',')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    try:
        normals = np.loadtxt(fname + '.normals', delimiter=',')
    except FileNotFoundError:
        print('Normals are mandatory. Estimating normals...')
        pcd.estimate_normals()
        pcd.normalize_normals()
        pcd.orient_normals_consistent_tangent_plane(k=20)
    else:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    trimesh = o3d.geometry.TriangleMesh()
    recon, _ = trimesh.create_from_point_cloud_poisson(pcd,
                                                       depth=depth,
                                                       scale=scale,
                                                       n_threads=n_threads)
    if save:
        o3d.io.write_triangle_mesh(fname + '.ply',
                                   recon,
                                   write_ascii=write_ascii,
                                   write_vertex_colors=False,
                                   print_progress=True)
        print(f'Reconstructed triangular mesh saved to `{fname}.ply`.')


if __name__ == '__main__':
    main()
