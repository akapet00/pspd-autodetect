import datetime
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import time

import numpy as np
try:
    import open3d as o3d
except ModuleNotFoundError as e:
    print(e, 'install it before proceeding', sep=', ')
else:
	o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(0))
from scipy import spatial
from tqdm.auto import tqdm

from .points import remove_hidden_points
from .normals import estimate_normals
from .misc import edblquad


class PSPD(object):
    """Automatic detection of the peak spatial power density."""
    def __init__(self, points, power_density, normals=None, mesh=None):
        """Constructor.
        
        Parameters
        ----------
        points : numpy.ndarray
            The point cloud of shape (N, 3), N is the number of points.
        power_density : numpy.ndarray
            Power density distribution. Either normalized or
            non-normalized. If normalized it is expected to be given in
            an array of shape (N, ). Otherwise, the shape should
            correspond to the shape of points where columns represent
            x-, y- and z-component of the (complex) power density.
        normals : numpy.ndarray, optional
            Normals of shape (N, 3), where N is the number of points in
            the point cloud. If mesh is not provided, normals should
            not be unit, i.e., non-normalized, as they will be used to
            estimate the surface area during the spatial averaging.
        mesh : open3d.geometry.TriangleMesh, optional
            Triangle mesh contains vertices and triangles represented
            by the indices to the vertices. Optionally, it also
            contains triangle and vertex normals and vertex colors.
        """
        # add logger
        self.log = logging.getLogger()

        # handle points
        size = points.shape[0]
        if size < 10:
            raise ValueError('Number of points must be > 10')
        else:
            self.size = size
        self.points = points
        
        # handle mesh - optional
        if isinstance(mesh, o3d.geometry.TriangleMesh):
            self.mesh = mesh
        else:
            self.log.info('Unrecognized mesh; proceeding without it...')
            self.mesh = None

        # handle normals
        if (normals is None) & (self.mesh is None):
            k = self._k
            self.log.info(f'Estimating normals with k-nn = {k}...')
            self.log.info(f'Execution started at {datetime.datetime.now()}')
            start_time = time.perf_counter()
            normals = estimate_normals(points,
                                       k,
                                       unit=False,
                                       orient=True)
            elapsed = time.perf_counter() - start_time
            self.log.info(f'Execution finished at {datetime.datetime.now()}')
            self.log.info(f'Elapsed time: {elapsed:.4f} s')
        self.normals = normals

        # handle absorbed or incident power density on the surface
        assert power_density.shape[0] == self.size, 'Size missmatch'
        if power_density.ndim == 1:  # surface-normal propagation-direction
            self.power_density_n = power_density
        elif power_density.ndim == 2:  # unoriented
            if power_density.shape[1] == 3:
                if self.normals is None:
                    k = self._k
                    self.log.info(f'Estimating normals with k-nn = {k}...')
                    self.log.info(f'Execution started at {datetime.datetime.now()}')
                    start_time = time.perf_counter()
                    normals = estimate_normals(points,
                                               k,
                                               unit=False,
                                               orient=True)
                    elapsed = time.perf_counter() - start_time
                    self.log.info(f'Execution finished at {datetime.datetime.now()}')
                    self.log.info(f'Elapsed time: {elapsed:.4f} s')
                    self.normals = normals
                self.power_density_n = np.sum(
                    np.real(power_density) * self.normals,
                    axis=1,
                )
            elif power_density.shape[2] == 1:
                self.power_density_n = np.ravel(power_density)
            else:
                raise ValueError('Unrecognized data distribution')
        else:
            raise ValueError('Only 1- and 2-D data supported')
        
        # dictionary for the results
        self.results = {'query point': [], 
                        'k-neigborhood': [],
                        'evaluation surface': [],
                        'surface area': [],
                        'power density': [],
                        'spatially averaged power density': []}
    
    def __str__(self):
        return f'Spatial domain with {self.size} points'

    def __repr__(self):
        return self.__str__()
    
    @property
    def _k(self):
        k = int(2 * np.log(self.size))
        if k < 5:
            k = 5
        elif k > 30:
            k = 30
        return k

    @property
    def _query_ball_radius(self):
        try:
            a = np.sqrt(self.projected_area)
        except AttributeError as e:
            print(e, 'projected area is not defined', sep=', ')
            return
        else:
            return np.sqrt(2) / 2 * a
    
    def _map(self, X, mapper=None):
        if mapper is None:
            C = X.T @ X
            mapper, _, _ = np.linalg.svd(C)
            return X @ mapper, mapper
        return X @ mapper
    
    def _bound_nbh(self, nbh, p):
        try:
            a = np.sqrt(self.projected_area)
        except AttributeError as e:
            print(e, 'projected area is not defined', sep=', ')
            return
        else:
            bbox = [p[0]-a/2, p[0]+a/2,
                    p[1]-a/2, p[1]+a/2]
            bbox_ind = np.where(
                (nbh[:, 0] >= bbox[0]) & (nbh[:, 0] <= bbox[1])
                & (nbh[:, 1] >= bbox[2]) & (nbh[:, 1] <= bbox[3])
            )[0]
            return bbox, bbox_ind
    
    def _bound_mesh(self, nbh_vert, bbox):
        bbox_ind = np.where(
            (nbh_vert[:, 0] >= bbox[0]) & (nbh_vert[:, 0] <= bbox[1])
            & (nbh_vert[:, 1] >= bbox[2]) & (nbh_vert[:, 1] <= bbox[3])
        )[0]
        return bbox_ind

    def _estimate_surf_area(self, domain):
        if isinstance(domain, np.ndarray):
            area = edblquad(domain[:, :2],
            				np.linalg.norm(domain[:, 2:], axis=1))
        elif isinstance(domain, o3d.geometry.TriangleMesh):
            area = domain.get_surface_area()
        else:
            print(NotImplementedError('Proceeding with the projected area'))
            area = self.projected_area
        return area


    def _step(self, p, rc):
        ind = self.tree.query_ball_point([p], rc)[0]
        nbh = self.points[ind]
        pdn = self.power_density_n[ind]
        if self.mesh:
            vind = self.vtree.query_ball_point([p], rc)[0]
            nbh_mesh = self.mesh.select_by_index(vind, cleanup=True)
            nbh_mesh = nbh_mesh.subdivide_midpoint(number_of_iterations=1)
            nbh_vert = np.asarray(nbh_mesh.vertices)
        else:  # use surface normals for surface area estimation
            n = self.normals[ind]
        
        # point cloud in the orthonormal basis
        mu = np.mean(nbh, axis=0)
        nbht, mapper = self._map(nbh - mu)
        pt = self._map(p - mu, mapper)

        # bounding box that corresponds to the projected surface
        bbox, nbh_bbox_ind = self._bound_nbh(nbht, pt)
        if self.mesh:
            nbht_vert = self._map(nbh_vert - mu, mapper)
            vert_bbox_ind = self._bound_mesh(nbht_vert, bbox)
            domain = nbh_mesh.select_by_index(vert_bbox_ind, cleanup=True)
        else:  # if mesh is not provided, domain is the local point cloud
            domain = nbht[nbh_bbox_ind, :2]
            domain = np.c_[domain, n[nbh_bbox_ind]]  # append surface normals
        
        # conformal surface area
        area = self._estimate_surf_area(domain)
        
        # spatially averaged absorbed power density
        spdn = 1 / area * edblquad(points=nbht[nbh_bbox_ind, :2],
                                   values=pdn[nbh_bbox_ind],
                                   bbox=bbox,
                                   s=1)
        
        # capture the rest of the values results
        nbh = nbh[nbh_bbox_ind]
        pdn = pdn[nbh_bbox_ind]
        return nbh, area, domain, pdn, spdn
            
    def find(self, projected_area, **kwargs):
        """Finds the peak spatially averaged power density on the
        non-planar surface.
        
        Parameters
        ----------
        projected_area : float
            Area of the square projection of the evaluation surface,
            units should correspond to units of the point cloud.
        kwargs : dict, optional
            Additional keyword arguments for
            `pspd.points.remove_hidden_points` function to restrict the
            the number of points and subsequently the search space.
        """
        self.projected_area = projected_area
        rc = self._query_ball_radius
        self.tree = spatial.KDTree(self.points)
        if self.mesh:
            self.vert = np.asarray(self.mesh.vertices)
            self.vtree = spatial.KDTree(self.vert)
        if kwargs:  # if exists, iterate only over "visible" set of points
            self.ind = remove_hidden_points(self.points, **kwargs)
        else:
            self.ind = ...
        self.points_visible = self.points[self.ind]
        self.log.info(f'Execution started at {datetime.datetime.now()}')
        start_time = time.perf_counter()
        for p in tqdm(self.points_visible):
            nbh, area, domain, pdn, spdn = self._step(p, rc)
            self.results['query point'].append(p)
            self.results['k-neigborhood'].append(nbh)
            self.results['surface area'].append(area)
            self.results['evaluation surface'].append(domain)
            self.results['power density'].append(pdn)
            self.results['spatially averaged power density'].append(spdn)
        elapsed = time.perf_counter() - start_time
        self.log.info(f'Execution finished at {datetime.datetime.now()}')
        self.log.info(f'Elapsed time: {elapsed:.4f} s')

    def get_results(self, peak=True):
        if peak:
            peak_results = dict()
            idx = np.argmax(self.results['spatially averaged power density'])
            for key in self.results.keys():
                peak_results[key] = self.results[key][idx]
            return peak_results
        return self.results
    
    def get_points(self, hidden=False):
        if hidden:
            return self.ind, self.points
        return self.ind, self.points_visible
