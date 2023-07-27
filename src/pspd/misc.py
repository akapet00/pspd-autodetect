import numpy as np
from scipy import integrate
from scipy import interpolate


def weightmat(p, nbh, kernel='linear', gamma=None):
    """Return scaling weights given distance between a targeted point
    and its surrounding local neighborhood.
    
    Parameters
    ----------
    p : numpy_ndarray
        Targeted point of shape (3, ).
    nbh : numpy.ndarray
        An array of shape (N, 3) representing the local neighborhood.
    kernel : str, optional
        The weighting function to use for MLS fitting. If not set, all
        weights will be set to 1.
    gamma : float, optional
        A scaling factor for the weighting function. If not given, it
        is set to 1.
        
    Returns
    -------
    numpy.ndarray
        Array with weights of (N, ).
    """
    dist = np.linalg.norm(nbh - p, axis=1)  # squared Euclidian distance
    if gamma is None:
        gamma = 1.
    if kernel == 'linear':
        w = np.maximum(1 - gamma * dist, 0)
    elif kernel == 'truncated':
        w = np.maximum(1 - gamma * dist ** 2, 0)
    elif kernel == 'inverse':
        w = 1 / (dist + 1e12) ** gamma
    elif kernel == 'gaussian':
        w = np.exp(-(gamma * dist) ** 2)
    elif kernel == 'multiquadric':
        w = np.sqrt(1 + (gamma * dist) ** 2)
    elif kernel == 'inverse_quadric':
        w = 1 / (1 + (gamma * dist) ** 2)
    elif kernel == 'inverse_multiquadric':
        w = 1 / np.sqrt(1 + (gamma * dist) ** 2 )
    elif kernel == 'thin_plate_spline':
        w = dist ** 2 * np.log(dist)
    elif kernel == 'rbf':
        w = np.exp(-dist ** 2 / (2 * gamma ** 2))
    elif kernel == 'cosine':
        w = (nbh @ p) / np.linalg.norm(nbh * p, axis=1)
    return w


def polyfit2d(x, y, z, deg=1, rcond=None, full_output=False):
    r"""Return the coefficients of a 2-D polynomial of a given degree.
    This function is the 2-D adapted version of `numpy.polyfit`.
    
    Note. The fitting assumes that the variable `z` corresponds the
    values of the "height" function such as z = f(x, y). The fitting is
    then performed on the polynomial given as:
    
    .. math:: f(x, y) = \sum_{i, j} c_{i, j} x^i y^j
    
    where `i + j Y= n` and `n` is the degree of a polynomial.
    
    Parameters
    ----------
    x, y : array_like, shape (N,)
        x- and y-oordinates of the M data points `(x[i], y[i])`.
    z : array_like, shape (M,)
        z-coordinates of the M data points.
    deg : int, optional
        Degree of the polynomial to be fit.
    rcond : float, optional
        Condition of the fit. See `scipy.linalg.lstsq` for details. All
        singular values less than `rcond` will be ignored.
    full_output : bool, optional
        Full diagnostic information from the SVD is returned if True,
        otherwise only the fitted coefficients are returned.
        
    Returns
    -------
    numpy.ndarray
        Array of coefficients of shape (deg+1, deg+1). If `full_output`
        is set to true, sum of the squared residuals of the fit, the
        effective rank of the design matrix, its singular values, and
        the specified value of `rcond` are also returned.
    """
    deg = int(deg)
    if deg < 1:
        raise ValueError('Degree must be at least 1.')
    # set up the Vandermode (design) matrix and the intercept vector
    A = np.polynomial.polynomial.polyvander2d(x, y, [deg, deg])
    b = z.flatten()
    # set up relative condition of the fit
    if rcond is None:
        rcond = x.size * np.finfo(x.dtype).eps
    # solve the least square
    coef, res, rank, s = np.linalg.lstsq(A, b, rcond=rcond)
    if full_output:
        return coef.reshape(deg+1, deg+1), res, rank, s
    return coef.reshape(deg+1, deg+1)


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
        if method is None:  # default settings - fast
            return f.integral(*bbox)
        if method == 'gauss':  # adaptive Gauss-Kronrad quadrature - slow
            from scipy import integrate
            f_wrap = lambda v, u: f(u, v)
            I, _ = integrate.dblquad(f_wrap, *bbox)
            return I
        else:  
            raise ValueError('Method is not supported')
