import argparse
import os

import numpy as np


SUPPORTED_SURF_RECON = ['advancing_front', 'poisson']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method',
                        type=str,
                        default='advancing_front',
                        choices=SUPPORTED_SURF_RECON,
                        help='method for surface reconstruction')
    parser.add_argument('--radius_ratio_bound',
                        type=float,
                        default=5,
                        help='scaler for the radius ration bound')
    parser.add_argument('--beta',
                        type=float,
                        default=0.52,
                        help='half the angle of the plausability wedge')
    parser.add_argument('--depth',
                        type=int,
                        default=8,
                        help='[poisson] maximum depth of the octree')
    parser.add_argument('--scale',
                        type=float,
                        default=1.1,
                        help='[poisson] input scaler to a uniform size')
    parser.add_argument('--save',
                        action='store_true',
                        help='save the reconstructed mesh')
    args = parser.parse_args()
    return args


def infer_knn(N):
    knn = int(2 * np.log(N))
    if knn < 5:
        knn = 5
    elif knn > 30:
        knn = 30
    return knn


def main():
    args = parse_args()
    method = args.method
    save = args.save
    fname = os.path.join('model', 'head.scaled')
    
    if method == 'advancing_front':
        from CGAL.CGAL_Kernel import Point_3
        from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
        from CGAL.CGAL_Point_set_3 import Point_set_3
        from CGAL.CGAL_Point_set_processing_3 import (
            estimate_global_k_neighbor_scale,
            compute_average_spacing,
            grid_simplify_point_set
        )
        from CGAL.CGAL_Advancing_front_surface_reconstruction import (
            advancing_front_surface_reconstruction
        )
        
        rrb = args.radius_ratio_bound
        beta = args.beta
        
        # point set loading & preprocessing
        points = Point_set_3(f'{fname}.xyz')
        k = estimate_global_k_neighbor_scale(points)  # k-neigborhood scale
        avg_space = compute_average_spacing(points, k)  # average point space
        grid_simplify_point_set(points, avg_space/2)
        
        # advancing front surface reconstruction
        polyhedron = Polyhedron_3()
        advancing_front_surface_reconstruction(points, polyhedron, rrb, beta)
        
        if save:
            polyhedron.write_to_file(f'{fname}.off')
            print(f'Reconstructed triangular mesh saved to `{fname}.off`.')
    elif method == 'poisson':
        try:
            import open3d as o3d
        except ImportError:
            print('open3d is required')
        depth = args.depth
        scale = args.scale
        
        points = np.loadtxt(fname + '.xyz')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        try:
            normals = np.loadtxt(fname + '.normals')
        except FileNotFoundError:
            knn = infer_knn(points.shape[0])
            print(f'Normals are mandatory. Estimating normals, knn = {knn}...')
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn))
            pcd.normalize_normals()
            pcd.orient_normals_consistent_tangent_plane(knn)
        else:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        trimesh = o3d.geometry.TriangleMesh()
        recon, _ = trimesh.create_from_point_cloud_poisson(pcd,
                                                           depth=depth,
                                                           scale=scale,
                                                           n_threads=-1)
        if save:
            o3d.io.write_triangle_mesh(fname + '.off',
                                       recon,
                                       write_ascii=True,
                                       write_vertex_colors=False,
                                       print_progress=True)
            print(f'Reconstructed triangular mesh saved to `{fname}.off`.')


if __name__ == '__main__':
    main()
