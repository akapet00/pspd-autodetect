import argparse
import os

from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from CGAL.CGAL_Polygon_mesh_processing import isotropic_remeshing
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_edge_length',
                        type=float,
                        default=0,
                        help='edge length targeted in the remeshed patch')
    parser.add_argument('--save',
                        action='store_true',
                        help='save the reconstructed mesh')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    edge_len = args.target_edge_length
    save = args.save
    fname = os.path.join('model', 'head.scaled')
    
    polyhedron = Polyhedron_3(f'{fname}.off')
    facets = []
    for f in polyhedron.facets():
        facets.append(f)
    isotropic_remeshing(facets, edge_len, polyhedron)

    if save:
        polyhedron.write_to_file(f'{fname}.iso.off')
        print(f'Remeshed triangular mesh saved to `{fname}.iso.off`.')


if __name__ == '__main__':
    main()
