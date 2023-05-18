import numpy as np

def generate_power_density(amplitude,
                           radius,
                           query_point,
                           points,
                           scaler=None):
    """Return a theoretical distribution of RF-EMF power in a Gaussian
    pattern.

    Note. Mathematical formulation taken from Foster et al., Health
    Physics 111(6):528-541, 2016, doi: 10.1097/HP.0000000000000571 

    Parameters
    ----------
    amplitude : float
        Peak incident power density at the centre of the irradiated
        region in W/m2.
    radius : float
        Radius of a circular area representing the irradiated region.
    query_point : numpy.ndarray
        Point at the centre of the irradiated region. Shape must be
        (3, ) where values correspond to x-, y-, and z-coordinate,
        respectively.
    points : numpy.ndarray
        All points of the irradiated region. Shape must be (N, 3) where
        columns correspond to x-, y-, and z-coordinate. N is the total
        number of points in the observed point cloud. Note that units
        of `radius`, `query_point` and `points` should match.
    scaler : list, optional
        Values that stretch the distance between the centre point and
        remaining points of the irradiated region component-wise. The
        length must be 3 where values correspond to scaler for x-, y-,
        and z-component, respectively.

    Return
    ------
    numpy.ndarray
        Spatial distribution of the incident power density of shape
        (N, ) where N is the total number of points in the observed
        point cloud.
    """
    assert isinstance(query_point, np.ndarray), 'must be numpy.ndarray'
    assert isinstance(points, np.ndarray), 'must be numpy.ndarray'
    if scaler and (len(scaler) == 3):
        scaler = np.asarray(scaler)
    else:
        scaler = np.array([1, 1, 1])
    distance = np.linalg.norm((points - query_point) / scaler, axis=1)
    power_density = amplitude * np.exp(-(distance / radius) ** 2)
    return power_density
