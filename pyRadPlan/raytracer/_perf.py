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


@njit
def _fast_compute_plane_alphas(
    dim_min: np.ndarray,
    dim_max: np.ndarray,
    planes: np.ndarray,
    source: np.ndarray,
    ray: np.ndarray,
) -> np.ndarray:
    num_rays = source.shape[0]
    dim_min = np.atleast_2d(dim_min).T
    dim_max = np.atleast_2d(dim_max).T

    # plane_grid = np.tile(planes, (num_rays, 1)) # not numba compatible
    plane_grid = np.repeat(planes, num_rays).reshape((-1, num_rays)).T
    plane_ix = np.atleast_2d(np.arange(0, len(planes)))

    # set invalid indices to NaN
    ix = (
        (plane_ix < dim_min)
        | (plane_ix > dim_max)
        | ((plane_ix == dim_min) & (plane_ix == dim_max))
        | np.isnan(dim_min)
        | np.isnan(dim_max)
    )
    # np.isnan()

    plane_grid.ravel()[ix.ravel()] = np.nan

    # repeat.reshape.T replaces np.tile for numba compatibility
    source_vec_t = np.repeat(source, plane_grid.shape[1]).reshape((-1, plane_grid.shape[1]))
    ray_vec_t = np.repeat(ray, plane_grid.shape[1]).reshape((-1, plane_grid.shape[1]))

    alphas = (plane_grid - source_vec_t) / ray_vec_t

    return alphas


@njit
def _fast_compute_all_alphas(
    source_points: np.ndarray,
    ray_vecs: np.ndarray,
    x_planes: np.ndarray,
    y_planes: np.ndarray,
    z_planes: np.ndarray,
    resolution: np.ndarray,
    precision: np.dtype,
) -> np.ndarray:
    num_rays = ray_vecs.shape[0]

    a_x_min = np.zeros(num_rays, dtype=precision)
    a_x_max = np.ones(num_rays, dtype=precision)
    a_x_ix = np.nonzero(ray_vecs[:, 0])
    if len(a_x_ix[0]) > 0:
        tmp = (
            np.column_stack(
                (
                    x_planes[0] - source_points[a_x_ix[0], 0],
                    x_planes[-1] - source_points[a_x_ix[0], 0],
                )
            )
            / ray_vecs[a_x_ix[0], 0, None]
        )
        for j in range(tmp.shape[0]):
            a_x_min[a_x_ix[0][j]] = np.nanmin(tmp[j])
            a_x_max[a_x_ix[0][j]] = np.nanmax(tmp[j])

    a_y_min = np.zeros(num_rays, dtype=precision)
    a_y_max = np.ones(num_rays, dtype=precision)
    a_y_ix = np.nonzero(ray_vecs[:, 1])
    if len(a_y_ix[0]) > 0:
        tmp = (
            np.column_stack(
                (
                    y_planes[0] - source_points[a_y_ix[0], 1],
                    y_planes[-1] - source_points[a_y_ix[0], 1],
                )
            )
            / ray_vecs[a_y_ix[0], 1, None]
        )
        for j in range(tmp.shape[0]):
            a_y_min[a_y_ix[0][j]] = np.nanmin(tmp[j])
            a_y_max[a_y_ix[0][j]] = np.nanmax(tmp[j])

    a_z_min = np.zeros(num_rays, dtype=precision)
    a_z_max = np.ones(num_rays, dtype=precision)
    a_z_ix = np.nonzero(ray_vecs[:, 2])
    if len(a_z_ix[0]) > 0:
        tmp = (
            np.column_stack(
                (
                    z_planes[0] - source_points[a_z_ix[0], 2],
                    z_planes[-1] - source_points[a_z_ix[0], 2],
                )
            )
            / ray_vecs[a_z_ix[0], 2, None]
        )
        for j in range(tmp.shape[0]):
            a_z_min[a_z_ix[0][j]] = np.nanmin(tmp[j])
            a_z_max[a_z_ix[0][j]] = np.nanmax(tmp[j])

    # Compute alpha_limits
    alpha_min_values = np.full(num_rays, np.nan, dtype=precision)
    alpha_max_values = np.full(num_rays, np.nan, dtype=precision)

    for i in range(num_rays):
        alpha_min_values[i] = np.nanmax(
            np.array([0, a_x_min[i], a_y_min[i], a_z_min[i]], dtype=precision)
        )
        alpha_max_values[i] = np.nanmin(
            np.array([1, a_x_max[i], a_y_max[i], a_z_max[i]], dtype=precision)
        )

    alpha_limits = np.column_stack((alpha_min_values, alpha_max_values))

    # Determine the direction of ray movement
    # ix_tp_bigger_sp = self._target_points > source_points
    # ix_tp_smaller_sp = self._target_points < source_points
    ix_tp_bigger_sp = ray_vecs > 0.0
    ix_tp_smaller_sp = ray_vecs < 0.0

    for d in range(3):  # Loop through x, y, z dimensions
        alpha_tmp = np.full((num_rays, 2), np.nan, dtype=precision)
        alpha_tmp[ix_tp_bigger_sp[:, d], 0] = alpha_limits[ix_tp_bigger_sp[:, d], 0]
        alpha_tmp[ix_tp_smaller_sp[:, d], 0] = alpha_limits[ix_tp_smaller_sp[:, d], 1]
        alpha_tmp[ix_tp_bigger_sp[:, d], 1] = alpha_limits[ix_tp_bigger_sp[:, d], 1]
        alpha_tmp[ix_tp_smaller_sp[:, d], 1] = alpha_limits[ix_tp_smaller_sp[:, d], 0]

        # Compute min and max indices (i_min, i_max, j_min, j_max, k_min, k_max)
        if d == 0:  # x dimension
            planes = x_planes
            res_d = resolution[0]
        elif d == 1:  # y dimension
            planes = y_planes
            res_d = resolution[1]
        else:  # z dimension
            planes = z_planes
            res_d = resolution[2]

        num_planes = planes.size

        dim_min = (
            num_planes
            - (planes[-1] - alpha_tmp[:, 0] * ray_vecs[:, d] - source_points[:, d]) / res_d
            - 1
        )
        dim_max = (source_points[:, d] + alpha_tmp[:, 1] * ray_vecs[:, d] - planes[0]) / res_d
        # Rounding
        dim_min = np.ceil(1 / 1000 * (np.round(1000 * dim_min)))
        dim_max = np.floor(1 / 1000 * (np.round(1000 * dim_max)))

        if d == 0:  # Assigning computed min and max indices to i_min, i_max
            i_min, i_max = dim_min, dim_max
        elif d == 1:  # Assigning computed min and max indices to j_min, j_max
            j_min, j_max = dim_min, dim_max
        else:  # Assigning computed min and max indices to k_min, k_max
            k_min, k_max = dim_min, dim_max

    # Compute alphas for each plane and merge parametric sets
    alpha_x = _fast_compute_plane_alphas(
        i_min,
        i_max,
        x_planes,
        source_points[:, 0],
        ray_vecs[:, 0],
    )
    alpha_y = _fast_compute_plane_alphas(
        j_min,
        j_max,
        y_planes,
        source_points[:, 1],
        ray_vecs[:, 1],
    )
    alpha_z = _fast_compute_plane_alphas(
        k_min,
        k_max,
        z_planes,
        source_points[:, 2],
        ray_vecs[:, 2],
    )

    alphas = np.concatenate((alpha_limits, alpha_x, alpha_y, alpha_z), axis=1)

    # Vectorized unique operation across rows
    # Sort alphas row-wise and remove duplicates
    alphas = np.sort(alphas, axis=1)  # Sort each row ascendingly
    mask = np.diff(alphas, axis=1) == 0  # Identify duplicates
    alphas[:, 1:][mask] = np.nan  # Replace duplicates with NaN
    alphas = np.sort(alphas, axis=1)

    # Size Reduction
    max_num_columns = np.max(np.sum(~np.isnan(alphas), axis=1))
    alphas = alphas[:, :max_num_columns]

    return alphas
