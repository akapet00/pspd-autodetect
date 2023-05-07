import numpy as np
from scipy import integrate
from scipy import interpolate
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


def edblquad(points, values, bbox=None, method=None, **kwargs):
    """Return the approximate solution to the double integral by
    observing sampled integrand function.
    
    Parameters
    ----------
    points : numpy.ndarray
        The point cloud of shape (N, 2), N is the number of points.
    values : numpy.ndarray
        Sampled integrand of shape (N, 3).
    bbox : list, optional
        Bounding box that defines integration domain.
    method : string, optional
        If None, the integral is computed by directly integrating
        splines. Alternative method is `gauss` which utilizes adaptive
        Gauss-Kronrad quadrature.
    kwargs : dict, optional
        Additional keyword arguments for
        `scipy.interpolate.SmoothBivariateSpline`.
    
    Returns
    -------
    float
        Approximation of the double integral.
    """
    if not isinstance(values, np.ndarray):
        raise Exception('`values` must be array-like.')
    try:
        if not bbox:
            bbox = [points[:, 0].min(), points[:, 0].max(),
                    points[:, 1].min(), points[:, 1].max()]
    except TypeError:
        print('`points` must be a 2-column array.')
    else:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f = interpolate.SmoothBivariateSpline(*points.T,
                                                  values,
                                                  bbox=bbox,
                                                  **kwargs)
        if method is None:  # default settings
            return f.integral(*bbox)
        if method == 'gauss':
            from scipy import integrate
            f_wrap = lambda v, u: f(u, v)
            I, _ = integrate.dblquad(f, *bbox)
            return I
        else:  
            raise ValueError('Method is not supported')
        return I
