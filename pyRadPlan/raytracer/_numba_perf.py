from numba import njit, prange
import numpy as np


@njit(parallel=True, nogil=True, cache=True)
def fast_spatial_circle_lookup(
    mesh_x: np.ndarray, mesh_z: np.ndarray, lookup_pos: np.ndarray, radius: float
) -> np.ndarray:
    """Lookup all points within a circle in a meshgrid.

    Parameters
    ----------
    mesh_x : np.ndarray
        The x-coordinates of the meshgrid (NxN)
    mesh_z : np.ndarray
        The y-coordinates of the meshgrid (NxN)
    lookup_pos : np.ndarray
        the reference positions to lookup (Mx3)
    radius : float
        The radius of the circle.

    Returns
    -------
    np.ndarray
        The indices of the points within the circle.
    """
    candidate_ray_mx = np.full(mesh_x.shape, False, dtype=np.bool_)
    for i in prange(lookup_pos.shape[0]):
        ix = (mesh_x.ravel() - lookup_pos[i, 0]) ** 2 + (
            mesh_z.ravel() - lookup_pos[i, 2]
        ) ** 2 <= radius**2
        candidate_ray_mx.ravel()[ix] = 1
    return candidate_ray_mx
