"""Frozen ray geometry from 84ebc92b for transform-preparation comparisons."""

import array_api_compat
from pyRadPlan.core.xp_utils.typing import Array
from pyRadPlan.core.xp_utils import cupy_available, pytorch_gpu_available
from pyRadPlan.dose.engines._base_pencilbeam import (
    logger,
    _calc_geo_dists_cupy_kernel,
    _calc_geo_dists_torch_kernel,
)


def _prepare_beam_geometry(self, beam_info: dict) -> dict:
    """Compact the final beam-valid coordinates once, after modality depth limits.

    Indices are ordered positions in the engine's selected dose voxels, not
    flattened dose-grid rows. The original beam arrays remain available to
    modality-specific initialization. Cache lifetime is one beam/shift.
    """
    xp = array_api_compat.array_namespace(beam_info["bev_coords"])
    indices = xp.nonzero(beam_info["valid_coords_all"])[0]
    return {
        "indices": indices,
        "coords": xp.take(beam_info["bev_coords"], indices, axis=0),
        "scenario_valid": [xp.take(mask, indices, axis=0) for mask in beam_info["valid_coords"]]
        if len(beam_info["valid_coords"]) > 1
        else None,
    }


def _get_ray_geometry_from_beam(self, ray: dict[str], beam_info: dict[str]):
    # Lazy preparation runs after the complete subclass _init_beam, including
    # particle depth limits. It is shared by all subsequent rays in this beam.
    if "_ray_geometry" not in beam_info:
        beam_info["_ray_geometry"] = self._prepare_beam_geometry(beam_info)
    prepared = beam_info["_ray_geometry"]
    xp = array_api_compat.array_namespace(prepared["coords"])
    lateral_ray_cutoff = self._get_lateral_distance_from_dose_cutoff_on_ray(ray)
    selected, radial_dist_sq, lat_dists, iso_lat_dists = self._calc_geo_dists_on_valid_coords(
        prepared["coords"],
        ray["source_point_bev"],
        ray["target_point_bev"],
        ray["sad"],
        float(lateral_ray_cutoff),
    )
    positions = xp.take(prepared["indices"], selected, axis=0)
    for key in (
        "ix",
        "radial_dist_sq",
        "lat_dists",
        "iso_lat_dists",
        "geo_depths",
        "rad_depths",
    ):
        ray[key] = []
    for scenario in range(len(beam_info["rad_depths"])):
        scenario_positions = positions
        radial, lateral, projected = radial_dist_sq, lat_dists, iso_lat_dists
        if prepared["scenario_valid"] is not None:
            valid = xp.take(prepared["scenario_valid"][scenario], selected, axis=0)
            within_ray = xp.nonzero(valid)[0]
            scenario_positions = xp.take(positions, within_ray, axis=0)
            radial = xp.take(radial_dist_sq, within_ray, axis=0)
            lateral = xp.take(lat_dists, within_ray, axis=0)
            projected = xp.take(iso_lat_dists, within_ray, axis=0)
        ray["ix"].append(xp.take(self._vdose_grid, scenario_positions, axis=0))
        ray["radial_dist_sq"].append(radial)
        ray["lat_dists"].append(lateral)
        ray["iso_lat_dists"].append(projected)
        for key in ("geo_depths", "rad_depths"):
            ray[key].append(xp.take(beam_info[key][scenario], scenario_positions, axis=0))


def _calc_geo_dists_on_valid_coords(
    self,
    rot_coords_subset: Array,
    source_point_bev: Array,
    target_point_bev: Array,
    sad: float,
    lateral_cutoff: float,
):
    """Return ordered local indices and distances for compact beam coordinates."""
    xp = array_api_compat.array_namespace(rot_coords_subset, source_point_bev, target_point_bev)
    if hasattr(xp, "linalg"):
        norm = xp.linalg.vector_norm
        cross = xp.linalg.cross
    else:

        def norm(arr: Array) -> Array:
            return xp.sqrt(xp.sum(arr**2, axis=-1))

        def cross(x: Array, y: Array) -> Array:
            return xp.stack(
                [
                    x[..., 1] * y[..., 2] - x[..., 2] * y[..., 1],
                    x[..., 2] * y[..., 0] - x[..., 0] * y[..., 2],
                    x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0],
                ],
                axis=-1,
            )

    # Make sure that the source-point is a 1D array with matching dtype
    source_point_bev = xp.astype(xp.reshape(source_point_bev, (3,)), rot_coords_subset.dtype)
    target_point_bev = xp.astype(xp.reshape(target_point_bev, (3,)), rot_coords_subset.dtype)

    # Put [0 0 0] position in the source point for beamlet who passes through isocenter
    a = -source_point_bev

    # Normalize the vector
    a /= norm(a)

    # Put [0 0 0] position in the source point for a single beamlet
    b = target_point_bev - source_point_bev

    # Normalize the vector
    b /= norm(b)

    # Calculate Rotation Matrix
    derived_rot_mat = None
    # Check for alignment with tolerance (avoid numerical instability/noise)
    if not xp.all(xp.abs(a - b) < 1e-7):
        # Cross product
        cross_ab = cross(a, b)

        if self._eps_ijk is None:
            self._eps_ijk = xp.asarray(
                [
                    [[0, 0, 0], [0, 0, -1], [0, 1, 0]],
                    [[0, 0, 1], [0, 0, 0], [-1, 0, 0]],
                    [[0, -1, 0], [1, 0, 0], [0, 0, 0]],
                ],
                dtype=cross_ab.dtype,
                device=array_api_compat.device(cross_ab),
            )

        ssc_matrix = xp.tensordot(cross_ab, self._eps_ijk, axes=1)

        derived_rot_mat = (
            xp.eye(3, dtype=a.dtype, device=array_api_compat.device(a))
            + ssc_matrix
            + ssc_matrix @ ssc_matrix * (1 - xp.vecdot(a, b)) / (norm(cross_ab) ** 2)
        )

    lateral_cutoff_sq_over_sad_sq = (lateral_cutoff / sad) ** 2
    use_cupy = "cupy" in xp.__name__ and cupy_available() and xp.cuda.is_available()
    use_torch = "torch" in xp.__name__ and pytorch_gpu_available()
    if (use_cupy or use_torch) and rot_coords_subset.shape[0] > 0:
        if not self._cuda_kernel_used:
            logger.info("Using %s kernel for geometric distance calculation.", xp.__name__)
            self._cuda_kernel_used = True
        if derived_rot_mat is None:
            derived_rot_mat = xp.eye(3, dtype=a.dtype, device=self.device)
        derived_rot_mat = xp.astype(derived_rot_mat, rot_coords_subset.dtype)
        if use_cupy:
            # Retain the existing fused kernel and its arithmetic order.
            _, out_y, _, out_lat_x, out_lat_z, out_rad_dist_sq, out_mask = (
                _calc_geo_dists_cupy_kernel(
                    rot_coords_subset.ravel(),
                    derived_rot_mat.ravel(),
                    source_point_bev,
                    lateral_cutoff_sq_over_sad_sq,
                    size=rot_coords_subset.shape[0],
                )
            )
            out_lat_dists = xp.stack((out_lat_x, out_lat_z), axis=1)
        else:
            out_coords_temp, out_lat_dists, out_rad_dist_sq, out_mask = (
                _calc_geo_dists_torch_kernel(
                    rot_coords_subset,
                    derived_rot_mat,
                    source_point_bev,
                    float(lateral_cutoff_sq_over_sad_sq),
                )
            )
            out_y = out_coords_temp[:, 1]
    else:
        if not getattr(self, "_fallback_logged", False):
            logger.info("Using vectorized path for geometric distance calculation.")
            self._fallback_logged = True
        if derived_rot_mat is None:
            out_coords_temp = rot_coords_subset
        else:
            derived_rot_mat = xp.astype(derived_rot_mat, rot_coords_subset.dtype)
            out_coords_temp = xp.matmul(rot_coords_subset, derived_rot_mat)
        lx = out_coords_temp[:, 0] + source_point_bev[0]
        lz = out_coords_temp[:, 2] + source_point_bev[2]
        out_lat_dists = xp.stack((lx, lz), axis=1)
        out_rad_dist_sq = lx * lx + lz * lz
        out_y = out_coords_temp[:, 1]
        limit = lateral_cutoff_sq_over_sad_sq * out_y * out_y
        out_mask = out_rad_dist_sq <= limit

    # for array API compatible indexing
    sub_ix = xp.nonzero(out_mask)[0]

    # Return radial distances squared
    rad_distances_sq = xp.take(out_rad_dist_sq, sub_ix, axis=0)

    # Lateral distances in X & Z
    lat_dists = xp.take(out_lat_dists, sub_ix, axis=0)

    if array_api_compat.size(sub_ix) > 0:
        # Lateral distances projected onto isocenter
        iso_lat_dists = lat_dists / xp.take(out_y, sub_ix, axis=0)[:, None] * sad
    else:
        iso_lat_dists = xp.empty_like(lat_dists)

    return sub_ix, rad_distances_sq, lat_dists, iso_lat_dists
