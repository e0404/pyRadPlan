"""Frozen geometry reference from 4c5bef97 for performance/correctness comparisons.

The installer uses production globals, but retains the pre-refactor method bodies.
"""

from __future__ import annotations

# Frozen methods are compiled with production globals by install_reference.
# ruff: noqa: F821, PLR0912, PLR0915

import inspect
import sys


def reference_methods(cls):
    """Compile the frozen methods with production globals."""
    namespace = dict(sys.modules[cls.__module__].__dict__)
    for name in ("calc_geo_dists", "_get_ray_geometry_from_beam", "_compute_ssd", "_init_beam"):
        exec(
            compile(inspect.getsource(globals()[name]), "<geometry reference>", "exec"), namespace
        )
    return {
        name: namespace[name]
        for name in ("calc_geo_dists", "_get_ray_geometry_from_beam", "_compute_ssd", "_init_beam")
    }


def install_reference(cls):
    """Install the old geometry path into this benchmark process only."""
    for name, method in reference_methods(cls).items():
        setattr(cls, name, method)


def _init_beam(self, _dij: dict, ct: CT, _cst: StructureSet, stf: SteeringInformation, i) -> dict:
    """
    Initialize the beam for pencil beam dose calculation.

    Parameters
    ----------
    dij : dict
        The dose influence matrix dictionary.
    ct : CT
        The CT object.
    _cst : StructureSet
        The structure set object. Unused here
    stf : SteeringInformation
        The steering information object.
    i : int
        Index of the beam.

    Returns
    -------
    dict
        Beam Information dictionary
    """
    # TODO: here .model_dump() is used to get beam as dict. Its possible to change...
    # ... the ray_tracing to handle this as the model
    beam_info = {"beam": stf.beams[i].model_dump(), "beam_index": i}

    xp = array_api_compat.array_namespace(self._vox_world_coords)

    for key in [
        "iso_center",
        "source_point_bev",
        "source_point",
        "num_of_bixels_per_ray",
        "gantry_angle",
        "couch_angle",
    ]:
        if array_api_compat.is_array_api_obj(beam_info["beam"][key]):
            beam_info["beam"][key] = to_namespace(xp, beam_info["beam"][key], device=self.device)
        else:
            beam_info["beam"][key] = xp.asarray(
                beam_info["beam"][key], dtype=xp.float32, device=self.device
            )
    for ray in beam_info["beam"]["rays"]:
        for key in ["ray_pos", "ray_pos_bev", "target_point", "target_point_bev"]:
            if ray[key] is None:
                continue
            if array_api_compat.is_array_api_obj(ray[key]):
                ray[key] = to_namespace(xp, ray[key], device=self.device)
            else:
                ray[key] = xp.asarray(ray[key], dtype=xp.float32, device=self.device)

    # Convert voxel indices to real coordinates using iso center of beam i
    iso_center = xp.asarray(beam_info["beam"]["iso_center"], dtype=xp.float32, device=self.device)

    coords_v = self._vox_world_coords - iso_center
    coords_vdose_grid = self._vox_world_coords_dose_grid - iso_center

    # Get Rotation Matrix
    rot_mat = get_beam_rotation_matrix(
        beam_info["beam"]["gantry_angle"], beam_info["beam"]["couch_angle"]
    )
    beam_info["rot_mat_system_T"] = xp.asarray(rot_mat, dtype=xp.float32, device=self.device)

    # Rotate coordinates (1st couch around Y axis, 2nd gantry movement)
    rot_coords_v = xp.matmul(coords_v, beam_info["rot_mat_system_T"])
    rot_coords_vdose_grid = xp.matmul(coords_vdose_grid, beam_info["rot_mat_system_T"])

    source_point_bev = xp.asarray(
        beam_info["beam"]["source_point_bev"], dtype=xp.float32, device=self.device
    )
    rot_coords_v -= source_point_bev
    rot_coords_vdose_grid -= source_point_bev

    # Calculate geometric distances
    geo_dist_vdose_grid = [
        xp.sqrt(xp.sum(rot_coords_vdose_grid**2, axis=1)) for _ in range(ct.num_of_ct_scen)
    ]

    # Calculate radiological depth cube
    logger.info("Calculating radiological depth cube...")

    start_time = time.time()

    self._raytracer.debug_core_performance = True
    self._raytracer.lateral_cut_off = beam_info["beam"].get(
        "effective_lateral_cut_off", self._effective_lateral_cutoff
    )
    patient_coords = rot_coords_vdose_grid if self.trace_on_dose_grid else rot_coords_v
    rad_depth_cubes = self._raytracer.trace_cubes_array(
        beam_info["beam"], patient_coords=patient_coords
    )
    self._raytracer.debug_core_performance = False

    # TODO: add universal time-debugging method
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info("Elapsed time for rayTracing per Beam: %f seconds", elapsed_time)

    beam_info["valid_coords"] = [None] * len(rad_depth_cubes)
    beam_info["rad_depths"] = [None] * len(rad_depth_cubes)

    for c, rad_depth_vdose_grid in enumerate(self._sample_radiological_depths(rad_depth_cubes)):
        # Find valid coordinates
        coord_is_valid = xp.isfinite(rad_depth_vdose_grid)

        beam_info["valid_coords"][c] = coord_is_valid

        # TODO: !remove brackets once multscen is implemented
        beam_info["rad_depths"][c] = rad_depth_vdose_grid

    beam_info["geo_depths"] = geo_dist_vdose_grid
    beam_info["bev_coords"] = rot_coords_vdose_grid

    beam_info["valid_coords_all"] = xp.any(xp.stack(beam_info["valid_coords"]), axis=0)

    # Check existence of target_points
    if any(r["target_point"] is None for r in beam_info["beam"]["rays"]):
        logger.debug("Missing target_point in rays. Calculating target points...")
        for ray in beam_info["beam"]["rays"]:
            ray["target_point"] = 2 * ray["ray_pos"] - beam_info["beam"]["source_point"]
            ray["target_point_bev"] = (
                2 * ray["ray_pos_bev"] - beam_info["beam"]["source_point_bev"]
            )

    # Compute SSDs
    self._compute_ssd(beam_info, ct, density_threshold=self.ssd_density_threshold)

    logger.info("Done.")

    return beam_info


def _compute_ssd(
    self,
    beam_info: dict,
    _ct: CT,
    mode: Literal["first"] = "first",
    density_threshold: float = 0.05,
    show_warning: bool = True,
):
    """
    Compute SSD (Source to Surface Distance) for each ray.

    Parameters
    ----------
    beam_info : dict
        Beam Information, will be modified to include SSD.
    ct : CT
        The CT object.
    mode : Literal["first"], optional
        Mode for handling multiple cubes to compute one SSD. Only 'first' is implemented.
    density_threshold : float, optional
        Value determining the skin threshold.
    show_warning : bool, optional
        Flag to show warnings.
    """

    beam = beam_info["beam"]

    # TODO: Add MultiScenario Support - remove this line
    if mode == "first":
        ssd = [None] * beam["num_of_rays"]

        xp = array_api_compat.array_namespace(beam["iso_center"], beam["source_point"])

        # Build stacked arrays using the backend's stack (works for numpy, cupy, torch)
        ray_pos_bev = xp.stack([xp.asarray(ray["ray_pos_bev"]) for ray in beam["rays"]])
        target_points = xp.stack([xp.asarray(ray["target_point"]) for ray in beam["rays"]])

        alpha, _, rho, d12, _ = self._raytracer.trace_rays(
            beam["iso_center"],
            xp.reshape(beam["source_point"], (1, 3)),
            target_points,
        )

        alpha = to_namespace(xp, alpha, device=self.device)
        rho = [to_namespace(xp, r, device=self.device) for r in rho]
        d12 = to_namespace(xp, d12.squeeze(), device=self.device)
        # prevent d12 from being 0d array which causes issues with indexing later on
        if d12.ndim == 0:
            d12 = xp.reshape(d12, (1,))
        # find rays that do not hit patients
        ix_nan = xp.all(xp.isnan(rho[0]), axis=1)
        if xp.any(ix_nan):
            msg = (
                f"{xp.count_nonzero(ix_nan)} rays do not hit patient. Trying to fix afterwards..."
            )
            if show_warning:
                warnings.warn(msg)
            else:
                logger.warning(msg)

        ix_ssd = -1 * xp.ones_like(ix_nan, dtype=xp.int64)

        # Work with explicit row indices to avoid unsupported boolean argmax on some backends
        row_idx = xp.nonzero(~ix_nan)[0]

        # Compute boolean condition, convert to integers for argmax
        cond = xp.take(rho[0], row_idx, axis=0) > density_threshold
        cond_int = xp.where(cond, xp.asarray(1, dtype=xp.int64), xp.asarray(0, dtype=xp.int64))
        arg = xp.argmax(cond_int, axis=1)

        ix_ssd = xp.where(~ix_nan, xp.astype(arg, xp.int64), ix_ssd)

        ssd = beam["sad"] * xp.ones_like(ix_ssd, dtype=xp.float32)

        d12_sel = xp.reshape(xp.take(d12, row_idx, axis=0), (-1,))
        alpha_rows = xp.take(alpha, row_idx, axis=0)
        idx = xp.reshape(xp.astype(arg, xp.int64), (-1, 1))
        alpha_sel = xp.reshape(xp.take_along_axis(alpha_rows, idx, axis=1), (-1,))

        ssd_vals = xp.astype(d12_sel, xp.float32) * xp.astype(alpha_sel, xp.float32)
        ssd = xp.where(~ix_nan, ssd_vals, ssd)

        # Now assign the ssd to all rays
        for j, ray in enumerate(beam["rays"]):
            if ix_nan[j]:  # Try to fix SSD by using SSD of closest neighboring ray
                ray["SSD"] = self._closest_neighbor_ssd(ray_pos_bev, ssd, ray["ray_pos_bev"])
            else:
                ray["SSD"] = float(ssd[j])

    else:
        raise ValueError(f"Invalid mode {mode} for SSD calculation")

    beam_info["beam"] = beam


def _get_ray_geometry_from_beam(self, ray: dict[str], beam_info: dict[str]):
    lateral_ray_cutoff = self._get_lateral_distance_from_dose_cutoff_on_ray(ray)

    xp = array_api_compat.array_namespace(beam_info["bev_coords"])

    # Ray tracing for beam i and ray j
    ix, radial_dist_sq, lat_dists, iso_lat_dists = self.calc_geo_dists(
        beam_info["bev_coords"],
        ray["source_point_bev"],
        ray["target_point_bev"],
        ray["sad"],
        beam_info["valid_coords_all"],
        float(lateral_ray_cutoff),
    )
    # Subindex given the relevant indices from the geometric distance calculation
    ray["valid_coords"] = [beam_ix & ix for beam_ix in beam_info["valid_coords"]]
    ray["ix"] = [self._vdose_grid[ix_in_grid] for ix_in_grid in ray["valid_coords"]]

    ray["radial_dist_sq"] = [radial_dist_sq[beam_ix[ix]] for beam_ix in ray["valid_coords"]]

    if lat_dists is not None:
        ray["lat_dists"] = [lat_dists[beam_ix[ix]] for beam_ix in beam_info["valid_coords"]]

    ray["iso_lat_dists"] = [iso_lat_dists[beam_ix[ix]] for beam_ix in ray["valid_coords"]]

    ray["valid_coords_all"] = xp.any(xp.stack(ray["valid_coords"], axis=0), axis=1)

    ray["geo_depths"] = [
        rD[ix] for rD, ix in zip(beam_info["geo_depths"], ray["valid_coords"])
    ]  # usually not needed for particle beams
    ray["rad_depths"] = [rD[ix] for rD, ix in zip(beam_info["rad_depths"], ray["valid_coords"])]


def calc_geo_dists(
    self,
    rot_coords_bev: Array,
    source_point_bev: Array,
    target_point_bev: Array,
    sad: float,
    rad_depth_mask: Array,
    lateral_cutoff: float,
):
    """
    Calculate geometric distances for dose calculation.

    This function relies on FOUR different ways to compute distances:
    1) GPU with cupy kernel
    2) GPU with torch kernel
    3) Base path (CPU/GPU) - backend supports fancy indexing regarding __setitem__
        -> numpy/cupy/torch
    4) Fallback path - backend does not support fancy indexing regarding __setitem__
        -> jax/strict

    Parameters
    ----------
    rot_coords_bev : Array
        Coordinates in beam's eye view (BEV) of the voxels where ray tracing results are
        available.
    source_point_bev : Array
        Source point in voxel coordinates in BEV.
    target_point_bev : Array
        Target point in voxel coordinates in BEV.
    sad : float
        Source-to-axis distance.
    rad_depth_mask : Array
        Masks the voxels for which radiological depth calculations are available.
    lateral_cutoff : float
        Lateral cutoff specifying the neighborhood for dose calculations.

    Returns
    -------
    ix : Array
        Indices of voxels where dose influence is computed.
    rad_distances_sq : Array
        Squared radial distances to the central ray.
    lat_dists : Array
        Lateral distances to the central ray (in X & Z).
    iso_lat_dists : Array
        Lateral distances to the central ray projected onto the isocenter plane.
    """

    # xp = array_api_compat.array_namespace(
    #     rot_coords_bev, source_point_bev, target_point_bev, rad_depth_mask
    # )
    xp = array_api_compat.array_namespace(
        rot_coords_bev, source_point_bev, target_point_bev, rad_depth_mask
    )
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

    # We later need a copy of the index, and we do it here to do some
    # compatibility management with the array API standard
    ix = xp.asarray(rad_depth_mask, copy=True)
    rad_depth_ix = xp.nonzero(rad_depth_mask)[0]

    # Make sure that the source-point is a 1D array with matching dtype
    source_point_bev = xp.astype(xp.reshape(source_point_bev, (3,)), rot_coords_bev.dtype)
    target_point_bev = xp.astype(xp.reshape(target_point_bev, (3,)), rot_coords_bev.dtype)

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

    # GPU Path (CuPy)
    if (
        cupy_available()
        and hasattr(xp, "cuda")
        and xp.cuda.is_available()
        and "cupy" in xp.__name__
    ):
        if not self._cuda_kernel_used:
            # necessary to avoid multiple logging
            logger.info("Using Cupy kernel for geometric distance calculation.")
            self._cuda_kernel_used = True

        rot_coords_subset = xp.take(rot_coords_bev, rad_depth_ix, axis=0)
        num_elements = rot_coords_subset.shape[0]

        if num_elements > 0:
            if derived_rot_mat is None:
                derived_rot_mat = xp.eye(3, dtype=a.dtype, device=self.device)

            # Ensure derived_rot_mat is correct type
            derived_rot_mat = xp.astype(derived_rot_mat, rot_coords_subset.dtype)

            lateral_cutoff_sq_over_sad_sq = (lateral_cutoff / sad) ** 2

            # NOTE!: I tried using the RawKernel. However saw no increase in performance.
            # Thus, we stick to ElementwiseKernel for now (easier to maintain)

            # CuPy ElementwiseKernel: call as function, returns values
            # Inputs must be flattened for raw indexing
            # size= is required since all inputs are 'raw'
            (
                out_coords_x,
                out_coords_y,
                out_coords_z,
                out_lat_x,
                out_lat_z,
                out_rad_dist_sq,
                out_mask,
            ) = _calc_geo_dists_cupy_kernel(
                rot_coords_subset.ravel(),
                derived_rot_mat.ravel(),
                source_point_bev,
                lateral_cutoff_sq_over_sad_sq,
                size=num_elements,
            )

            # Stack outputs to match expected format
            out_coords_temp = xp.stack((out_coords_x, out_coords_y, out_coords_z), axis=1)
            out_lat_dists = xp.stack((out_lat_x, out_lat_z), axis=1)

            ix[rad_depth_ix] = out_mask

            # Filter results
            # Optimization: sync once for sub_ix
            sub_ix = xp.nonzero(out_mask)[0]
            rad_distances_sq = xp.take(out_rad_dist_sq, sub_ix, axis=0)
            lat_dists = xp.take(out_lat_dists, sub_ix, axis=0)

            if array_api_compat.size(sub_ix) > 0:
                iso_lat_dists = lat_dists / out_coords_temp[sub_ix, 1][:, None] * sad
            else:
                iso_lat_dists = xp.empty_like(lat_dists)

            return ix, rad_distances_sq, lat_dists, iso_lat_dists

    # GPU Path (PyTorch)
    if pytorch_gpu_available() and "torch" in xp.__name__:
        if not self._cuda_kernel_used:
            logger.info("Using PyTorch kernel for geometric distance calculation.")
            self._cuda_kernel_used = True

        rot_coords_subset = xp.take(rot_coords_bev, rad_depth_ix, axis=0)
        num_elements = rot_coords_subset.shape[0]

        if num_elements > 0:
            if derived_rot_mat is None:
                derived_rot_mat = xp.eye(3, dtype=a.dtype, device=self.device)

            # Ensure derived_rot_mat is correct type
            derived_rot_mat = xp.astype(derived_rot_mat, rot_coords_subset.dtype)

            lateral_cutoff_sq_over_sad_sq = (lateral_cutoff / sad) ** 2

            out_coords_temp, out_lat_dists, out_rad_dist_sq, out_mask = (
                _calc_geo_dists_torch_kernel(
                    rot_coords_subset,
                    derived_rot_mat,
                    source_point_bev,
                    float(lateral_cutoff_sq_over_sad_sq),
                )
            )
            # TODO: GPU synch happens here. Maybe this can be avoided?
            ix[rad_depth_ix] = out_mask

            # Filter results
            # Optimization: sync once for sub_ix
            sub_ix = xp.nonzero(out_mask)[0]
            rad_distances_sq = xp.take(out_rad_dist_sq, sub_ix, axis=0)
            lat_dists = xp.take(out_lat_dists, sub_ix, axis=0)

            if array_api_compat.size(sub_ix) > 0:
                iso_lat_dists = lat_dists / out_coords_temp[sub_ix, 1][:, None] * sad
            else:
                iso_lat_dists = xp.empty_like(lat_dists)

            return ix, rad_distances_sq, lat_dists, iso_lat_dists

    # Vectorized CPU/NumPy path (Fallback)
    if not self._cuda_kernel_used:
        if not getattr(self, "_fallback_logged", False):
            logger.info("Using vectorized path for geometric distance calculation.")
            self._fallback_logged = True

    rot_coords_subset = xp.take(rot_coords_bev, rad_depth_ix, axis=0)

    # 1. Matrix Multiplication (N,3) @ (3,3) -> (N,3)
    if derived_rot_mat is None:
        out_coords_temp = rot_coords_subset
    else:
        derived_rot_mat = xp.astype(derived_rot_mat, rot_coords_subset.dtype)
        if hasattr(xp, "matmul"):
            out_coords_temp = xp.matmul(rot_coords_subset, derived_rot_mat)
        else:
            out_coords_temp = rot_coords_subset @ derived_rot_mat

    # 2. Calculate Lateral Distances
    lx = out_coords_temp[:, 0] + source_point_bev[0]
    lz = out_coords_temp[:, 2] + source_point_bev[2]

    out_lat_dists = xp.stack((lx, lz), axis=1)

    # 3. Radial Distance Squared
    out_rad_dist_sq = lx * lx + lz * lz

    # 4. Mask
    lateral_cutoff_sq_over_sad_sq = (lateral_cutoff / sad) ** 2
    ry = out_coords_temp[:, 1]
    limit = lateral_cutoff_sq_over_sad_sq * ry * ry
    out_mask = out_rad_dist_sq <= limit

    # Check if backend supports fancy indexing (JAX and array_api_strict don't)
    xp_name = getattr(xp, "__name__", "")
    supports_fancy_setitem = "jax" not in xp_name and "strict" not in xp_name

    # Fancy indexing is supported for cupy/numpy/torch!
    # in some cases the kernels above are not compiled
    # So cupy and torch might rely on this path too.
    if supports_fancy_setitem:
        # Fast path: use fancy indexing scatter
        ix[rad_depth_ix] = out_mask
    else:
        # NOTE! This fallback is necessary for jax/strict. Above can be called for numpy/torch/cupy.
        # -> Above is approx 20~30% faster (proton dose calc)
        # Slow path for immutable arrays: compute mask on full array
        if derived_rot_mat is None:
            out_coords_full = rot_coords_bev
        elif hasattr(xp, "matmul"):
            out_coords_full = xp.matmul(rot_coords_bev, derived_rot_mat)
        else:
            out_coords_full = rot_coords_bev @ derived_rot_mat
        lx_full = out_coords_full[:, 0] + source_point_bev[0]
        lz_full = out_coords_full[:, 2] + source_point_bev[2]
        rad_dist_sq_full = lx_full * lx_full + lz_full * lz_full
        ry_full = out_coords_full[:, 1]
        limit_full = lateral_cutoff_sq_over_sad_sq * ry_full * ry_full
        cutoff_mask_full = rad_dist_sq_full <= limit_full
        ix = rad_depth_mask & cutoff_mask_full

    # for array API compatible indexing
    sub_ix = xp.nonzero(out_mask)[0]

    # Return radial distances squared
    rad_distances_sq = xp.take(out_rad_dist_sq, sub_ix, axis=0)

    # Lateral distances in X & Z
    lat_dists = xp.take(out_lat_dists, sub_ix, axis=0)

    if array_api_compat.size(sub_ix) > 0:
        # Lateral distances projected onto isocenter
        iso_lat_dists = lat_dists / out_coords_temp[sub_ix, 1][:, None] * sad
    else:
        iso_lat_dists = xp.empty_like(lat_dists)

    return ix, rad_distances_sq, lat_dists, iso_lat_dists
