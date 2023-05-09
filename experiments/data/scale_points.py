import argparse
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scaler',
                        type=float,
                        default=100,
                        help='apply scaler to all points')
    parser.add_argument('--save',
                        action='store_true',
                        help='save estimated normals')
    args = parser.parse_args()
    return args
    

def main():
    args = parse_args()
    scaler = args.scaler
    save = args.save
    fname = os.path.join('model', 'head')
    
    points = np.loadtxt(f'{fname}.xyz')
    points_scaled = points * scaler
    
    if save:
        np.savetxt(f'{fname}.scaled.xyz', points_scaled)
        print(f'Scaled points saved to `{fname}.scaled.xyz`.')


if __name__ == '__main__':
    main()
