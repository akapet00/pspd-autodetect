import numpy as np
import open3d as o3d
from scipy import spatial

from .misc import polyfit2d
from .misc import weightmat


def orient_normals(points, normals, k):
    """Orient the normals with respect to consistent tangent planes.
    
    Ref: Hoppe et al., in proceedings of SIGGRAPH 1992, pp. 71-78,
         doi: 10.1145/133994.134011
    
    Parameters
    ----------
    points : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
    normals : numpy.ndarray
        Normals of shape (N, 3), where N is the number of points in the
        point cloud.
    k : int
        Number of k nearest neighbors used in constructing the
        Riemannian graph used to propagate normal orientation.
    
    Returns
    -------
    numpy.ndarray
        Oriented normals.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.orient_normals_consistent_tangent_plane(k)
    return np.asarray(pcd.normals)


def estimate_normals(points,
                     k,
                     deg=1,
                     unit=True,
                     kernel=None,
                     orient=False,
                     **kwargs):
    """Return the (unit) normals by fitting 2-D polynomial at each
    point in the point cloud considering its local neighborhood.
    
    Parameters
    ----------
    points : numpy.ndarray
        The point cloud of shape (N, 3), N is the number of points.
    k : float
        The number of nearest neighbors of a local neighborhood around
        a current query point.
    deg : float, optional
        Degrees of the polynomial.
    unit : float, optional
        If true, normals are normalized. Otherwise, surface normals are
        returned.
    kernel : string, optional
        Kernel for computing distance-based weights.
    kwargs : dict, optional
        Additional keyword arguments for computing weights. For details
        see `weightmat` function.
    
    Returns
    -------
    numpy.ndarray
        The (unit) normals of shape (N, 3), where N is the number of
        points in the point cloud.
    """
    # create a kd-tree for quick nearest-neighbor lookup
    normals = np.empty_like(points)
    tree = spatial.KDTree(points)
    for i, p in enumerate(points):
        _, idx = tree.query([p], k=k, eps=0.1, workers=-1)
        nbhd = points[idx.flatten()]
        
        # change the basis of the local neighborhood
        X = nbhd.copy()
        X = X - X.mean(axis=0)
        C = (X.T @ X) / (nbhd.shape[0] - 1)
        U, _, _ = np.linalg.svd(C)
        X_t = X @ U
        
        # compute weights given specific distance function
        if kernel:
            w = weightmat(p, nbhd, kernel, **kwargs)
        else:
            w = np.ones((nbhd.shape[0], ))
            
        # fit parametric surface by usign a (weighted) 2-D polynomial
        X_t_w = X_t * w[:, np.newaxis]
        c = polyfit2d(*X_t_w.T, deg=deg)
        
        # compute normals as partial derivatives of the "height" function
        cu = np.polynomial.polynomial.polyder(c, axis=0)
        cv = np.polynomial.polynomial.polyder(c, axis=1)
        ni = np.array([-np.polynomial.polynomial.polyval2d(*X_t_w[0, :2], cu),
                       -np.polynomial.polynomial.polyval2d(*X_t_w[0, :2], cv),
                       1])
        
        # convert normal coordinates into the original coordinate frame
        ni = U @ ni
        
        # normalize normals by considering the magnitude of each
        if unit:
            ni = ni / np.linalg.norm(ni, 2)
        normals[i, :] = ni
    if orient:
        normals = orient_normals(points, normals, k)
    return normals
