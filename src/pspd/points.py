import numpy as np
from scipy import spatial


def remove_hidden_points(xyz, pov, p=np.pi):
    """Return only the points of a given point cloud that are directly
    visible from a preset point of view.
    
    Ref: Katz et al. ACM Transactions on Graphics 26(3), pp: 24-es
         doi: 10.1145/1276377.1276407
    
    Parameters
    ----------
    xyz : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
    pov : numpy.ndarray
        Point of view for hidden points removal of shape (1, 3).
    p : float
        Parameter for the radius of the spherical transformation.
    
    Returns
    -------
    numpy.ndarray
        Indices of the directly visible points in a point cloud.
    """
    xyzt = xyz - pov  # move pov to the origin
    norm = np.linalg.norm(xyzt, axis=1)[:, np.newaxis]
    R = norm.max() * 10 ** p
    xyzf = xyzt + 2 * (R - norm) * (xyzt / norm) # perform spherical flip
    hull = spatial.ConvexHull(np.append(xyzf, [[0,0,0]], axis=0))
    return hull.vertices[:-1]
