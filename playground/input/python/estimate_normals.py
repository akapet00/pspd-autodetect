import argparse
import os

from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Point_set_3 import Point_set_3
from CGAL.CGAL_Point_set_processing_3 import *
import numpy as np


SUPPORTED_NORMAL_EST = ['jet', 'pca']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_nearest_neighbors',
                        type=int,
                        default=30,
                        help='number of nearest neighbors to be searched')
    parser.add_argument('--method',
                        type=str,
                        default='pca',
                        choices=SUPPORTED_NORMAL_EST,
                        help='method for normal estimation')
    parser.add_argument('--save',
                        action='store_true',
                        help='save estimated normals')
    args = parser.parse_args()
    return args
    

def main():
    args = parse_args()
    knn = args.k_nearest_neighbors
    method = args.method
    save = args.save
    fname = os.path.join('data', 'head.scaled')
    
    # point set loading
    points = Point_set_3(f'{fname}.xyz')
    
    # normal estimation
    if method == 'jet':
        jet_estimate_normals(points, knn)
    elif method == 'pca':
        pca_estimate_normals(points, knn)
    else:
        pass
    
    # normal orientation (otward pointing normals)
    mst_orient_normals(points, knn)
    
    if save:
        normals = []
        for n in points.normals():
            normals.append([n.x(), n.y(), n.z()])
        normals = np.array(normals)
        np.savetxt(f'{fname}.normals', normals)
        print(f'Estimated normals saved to `{fname}.normals`.')


if __name__ == '__main__':
    main()
