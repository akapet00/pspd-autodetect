import argparse
import os

import numpy as np
import open3d as o3d
try:
    import pymeshfix
except ImportError:
    method = 'manual'
else:
    method = 'auto'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save',
                        action='store_true',
                        help='save the reconstructed mesh')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    save = args.save
    fname = os.path.join('data', 'head.scaled.iso')
    mesh = o3d.io.read_triangle_mesh(fname + '.off')
    if mesh.is_watertight():
        print('Mesh is watertight. Skipping the execution...')
        return
    else:  # repair the mesh
        vert = np.asarray(mesh.vertices)
        tri = np.asarray(mesh.triangles)
        if method == 'auto':  # by using pymeshfix - fast
            vert_fix, tri_fix = pymeshfix.clean_from_arrays(vert, tri)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vert_fix)
            mesh.triangles = o3d.utility.Vector3iVector(tri_fix)
        elif method == 'manual':  # by using open3d - slow
            if not mesh.is_edge_manifold(allow_boundary_edges=True):
                print('Removing non-manifold edges...')
                mesh = mesh.remove_non_manifold_edges()
                print('Removing unreferenced vertices...')
                mesh.remove_unreferenced_vertices()
                print('Done')
            if not mesh.is_vertex_manifold():
                print('Removing non-manifold vertices...')
                ind = mesh.get_non_manifold_vertices()
                mesh.remove_vertices_by_index(np.asarray(ind))
                print('Done')
            if not mesh.is_orientable():
                print('Orienting triangles...')
                _ = mesh.orient_triangles()
                print('Done')
            if mesh.is_self_intersecting():
                print('Removing self-intersecting trianges...')
                ind = mesh.get_self_intersecting_triangles()
                mesh.remove_triangles_by_index(np.unique(ind))
                print('Removing unreferenced vertices...')
                mesh = mesh.remove_unreferenced_vertices()
                print('Removing non-manifold vertices...')
                ind = mesh.get_non_manifold_vertices()
                mesh.remove_vertices_by_index(ind)
                print('Done')
        else:
            pass

    if save:
        o3d.io.write_triangle_mesh(fname + '.watertight.off',
                                   mesh,
                                   write_ascii=True,
                                   write_vertex_colors=False,
                                   print_progress=True)
        print(f'Watertight mesh saved to `{fname}.watertight.off`.')


if __name__ == '__main__':
    main()
