"""
Collapsed cone convolution: kernel handling and energy transport.

Port of the matRad collapsed cone implementation (``matRad_fitCollapsedConeKernel`` and
``matRad_collapsedConeTransportBatched``). Kernel loading and fitting run once on the host
with NumPy and SciPy; TERMA and transport use Array API operations only and run unchanged
on NumPy, CuPy, PyTorch and array-api-strict.

References
----------
.. [1] Ahnesjo, Med. Phys. 16, 577 (1989)
.. [2] Bortfeld, Schlegel, Rhein, Med. Phys. 20, 311 (1993)
"""

import logging

import array_api_compat
import numpy as np

from pyRadPlan.core.xp_utils import erf, interp1d, scatter, to_numpy
from pyRadPlan.core.xp_utils.typing import Array
from pyRadPlan.machines import PhotonCollapsedConeKernel
from pyRadPlan.machines.photons._collapsed_cone_fit import _fminbnd

logger = logging.getLogger(__name__)


def cone_directions(
    kernel: PhotonCollapsedConeKernel, num_azimuthal_angles: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cone directions in the local frame (polar axis along z) and their kernel components.

    Parameters
    ----------
    kernel : PhotonCollapsedConeKernel
        The kernel.
    num_azimuthal_angles : int
        Number of azimuthal cones per polar bin.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Directions ``(3, n_dirs)``, weights and attenuation coefficients
        ``(n_components, n_dirs)``. The weights are divided by the number of azimuthal cones.
    """
    phi = (np.arange(1, num_azimuthal_angles + 1) - 0.5) * 360.0 / num_azimuthal_angles
    # Column-major flattening as in matRad: the polar angle runs fastest
    theta, phi = np.meshgrid(np.deg2rad(kernel.polar_angles), np.deg2rad(phi), indexing="xy")
    theta = theta.ravel()
    phi = phi.ravel()
    dirs = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    weights = np.tile(kernel.weights, (1, num_azimuthal_angles)) / num_azimuthal_angles
    att = np.tile(kernel.attenuations, (1, num_azimuthal_angles))
    return dirs, weights, att


def tilt_directions(
    local_dirs: np.ndarray, axis_world: np.ndarray, lateral_world: np.ndarray
) -> np.ndarray:
    """
    Rotate local cone directions so that their polar axis is along ``axis_world``.

    Parameters
    ----------
    local_dirs : np.ndarray
        ``(3, n_dirs)`` directions with the polar axis along z.
    axis_world : np.ndarray
        Polar axis in world coordinates.
    lateral_world : np.ndarray
        Lateral direction in world coordinates, which fixes the azimuthal orientation.

    Returns
    -------
    np.ndarray
        Directions in world coordinates (x, y, z), shape ``(3, n_dirs)``.
    """
    u = np.asarray(axis_world, dtype=np.float64).ravel()
    u = u / np.linalg.norm(u)
    lateral = np.asarray(lateral_world, dtype=np.float64).ravel()
    e1 = lateral - np.dot(lateral, u) * u
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    return np.column_stack([e1, e2, u]) @ local_dirs


def calc_terma(
    sad: float,
    mu: float,
    sigma: float,
    bixel_width: float,
    primary_fluence: np.ndarray,
    ray_pos: np.ndarray,
    rad_depths: Array,
    geo_dists: Array,
    iso_lat_dists: Array,
    beta: float = 0.0,
) -> Array:
    """
    TERMA of a bixel per unit bixel weight.

    The depths and distances are arrays of one namespace and device; the TERMA is computed
    there in double precision.

    Parameters
    ----------
    sad : float
        Source to axis distance [mm].
    mu : float
        (Surface) effective attenuation coefficient [1/mm].
    sigma : float
        Penumbra sigma at the isocenter [mm].
    bixel_width : float
        Bixel width at the isocenter [mm].
    primary_fluence : np.ndarray
        Radial primary fluence profile ``(n, 2)`` [r, value].
    ray_pos : np.ndarray
        Ray position in beam's eye view [mm].
    rad_depths, geo_dists : Array
        Radiological depths and distances from the source [mm].
    iso_lat_dists : Array
        Lateral distances ``(n, 2)`` (x, z) to the ray in the isocenter plane [mm].
    beta : float
        Hardening coefficient [1/mm] of the attenuation ``mu / (1 + beta d)``.

    Returns
    -------
    Array
        TERMA per unit bixel weight on the voxels.
    """
    xp = array_api_compat.array_namespace(rad_depths, geo_dists, iso_lat_dists)
    device = array_api_compat.device(rad_depths)
    dtype = xp.float64
    rad_depths = xp.reshape(xp.astype(rad_depths, dtype, copy=False), (-1,))
    geo_dists = xp.reshape(xp.astype(geo_dists, dtype, copy=False), (-1,))
    iso_lat_dists = xp.reshape(xp.astype(iso_lat_dists, dtype, copy=False), (-1, 2))
    fluence = xp.asarray(np.asarray(primary_fluence, dtype=np.float64), device=device)
    s2 = float(np.sqrt(2) * sigma)
    half_width = 0.5 * float(bixel_width)

    # Lateral fluence of the bixel: a square of bixel_width convolved with the Gaussian
    # penumbra, along both lateral axes
    def edge(u):
        return 0.5 * (erf((u + half_width) / s2) - erf((u - half_width) / s2))

    # Beyond 10 sigma from the bixel edge the penumbra factor is below 1e-23: the expensive
    # erf terms are only evaluated on the voxels within that range
    reach = half_width + 10 * float(sigma)
    lat_x = iso_lat_dists[:, 0]
    lat_z = iso_lat_dists[:, 1]
    sel = xp.nonzero((xp.abs(lat_x) < reach) & (xp.abs(lat_z) < reach))[0]
    lat_x = xp.take(lat_x, sel)
    lat_z = xp.take(lat_z, sel)
    psi = edge(lat_x) * edge(lat_z)
    r = xp.sqrt((float(ray_pos[0]) + lat_x) ** 2 + (float(ray_pos[2]) + lat_z) ** 2)
    # Radial primary fluence of the machine at the voxel's position in the isocenter plane
    psi = psi * interp1d(r, fluence[:, 0], fluence[:, 1], left=0.0, right=0.0)

    # Inverse square law and depth dependence of the TERMA; voxels without a finite
    # distance or depth get no TERMA
    geo = xp.take(geo_dists, sel)
    depth = xp.take(rad_depths, sel)
    valid = xp.isfinite(geo) & (geo > 0) & xp.isfinite(depth)
    geo = xp.where(valid, geo, 1.0)
    depth = xp.where(valid, depth, 0.0)
    values = psi * (float(sad) / geo) ** 2 * terma_depth_factor(mu, beta, depth)
    values = xp.where(valid & xp.isfinite(values), values, 0.0)
    terma = xp.zeros(rad_depths.shape, dtype=dtype, device=device)
    return scatter(terma, sel, values)


def terma_depth_factor(mu: float, beta: float, depths: Array) -> Array:
    """
    TERMA per unit primary fluence in water for the attenuation ``mu / (1 + beta d)``.

    Parameters
    ----------
    mu : float
        Attenuation coefficient at the surface [1/mm].
    beta : float
        Hardening coefficient [1/mm], 0 for a constant attenuation.
    depths : Array
        Radiological depths [mm].

    Returns
    -------
    Array
        The TERMA factor at the depths.
    """
    xp = array_api_compat.array_namespace(depths)
    if beta > 0:
        base = 1 + beta * depths
        return mu / base * base ** (-mu / beta)
    return mu * xp.exp(-mu * depths)


def analytical_depth_dose(
    kernel_data_at_ssd: np.ndarray,
    kernel_pos: np.ndarray,
    betas: np.ndarray,
    m: float,
    depths: np.ndarray,
    conv_resolution: float = 0.5,
) -> np.ndarray:
    """
    Broad field central axis depth dose of the SVD pencil beam data (no inverse square).

    The kernels are integrated on the discrete grid of the SVD pencil beam engine, which
    resolves their central peaks.

    Parameters
    ----------
    kernel_data_at_ssd : np.ndarray
        Kernel components ``(n_components, n_pos)`` [1/mm^2] (version 2 convention).
    kernel_pos : np.ndarray
        Radial kernel positions [mm].
    betas : np.ndarray
        Build-up coefficients of the components [1/mm].
    m : float
        Attenuation coefficient [1/mm].
    depths : np.ndarray
        Depths [mm].
    conv_resolution : float
        Resolution [mm] of the integration grid, as ``int_conv_resolution`` of the SVD
        pencil beam engine.

    Returns
    -------
    np.ndarray
        Depth dose at ``depths``.
    """
    kernel_pos = np.asarray(kernel_pos, dtype=np.float64).ravel()
    limit = int(np.ceil(kernel_pos[-1] / conv_resolution))
    grid = np.arange(-limit, limit) * conv_resolution
    x, z = np.meshgrid(grid, grid)
    r = np.sqrt(x.ravel() ** 2 + z.ravel() ** 2)
    depths = np.asarray(depths, dtype=np.float64)
    dd = np.zeros_like(depths)
    for i, beta in enumerate(np.asarray(betas, dtype=np.float64).ravel()):
        integral = conv_resolution**2 * np.sum(
            np.interp(r, kernel_pos, kernel_data_at_ssd[i], left=0.0, right=0.0)
        )
        dd = dd + integral * beta / (beta - m) * (np.exp(-m * depths) - np.exp(-beta * depths))
    return dd


def broad_field_depth_dose(
    kernel: PhotonCollapsedConeKernel, mu: float, depths: np.ndarray, dz: float, beta: float = 0.0
) -> np.ndarray:
    """
    Collapsed cone broad field central axis depth dose in water (no inverse square).

    Computed with the recursion of the transport on a one dimensional depth grid: in a broad
    field every cone of a polar bin sees the same TERMA along its line, so the azimuthal
    cones collapse onto the depth axis with the step ``dz / |cos(theta)|``.

    Parameters
    ----------
    kernel : PhotonCollapsedConeKernel
        The kernel.
    mu : float
        Attenuation coefficient at the surface [1/mm].
    depths : np.ndarray
        Depths [mm].
    dz : float
        Depth resolution [mm].
    beta : float
        Hardening coefficient [1/mm].

    Returns
    -------
    np.ndarray
        Depth dose at ``depths``.
    """
    depths = np.asarray(depths, dtype=np.float64)
    num_vox = int(np.ceil((np.max(depths) + 300) / dz))
    z = (np.arange(1, num_vox + 1) - 0.5) * dz
    terma = terma_depth_factor(mu, beta, z)
    cos_polar = np.cos(np.deg2rad(kernel.polar_angles))
    weights = kernel.weights
    att = kernel.attenuations
    steps = dz / np.abs(cos_polar)
    dose = np.zeros(num_vox)
    for forward in (True, False):
        sel = (cos_polar > 0) == forward
        if not np.any(sel):
            continue
        w = weights[:, sel]
        x = att[:, sel] * steps[sel]
        one_minus_e = -np.expm1(-x)
        g = one_minus_e / x
        flow = np.zeros_like(w)
        order = range(num_vox) if forward else range(num_vox - 1, -1, -1)
        # Ahnesjo's recursion along the depth axis for the forward and backward cones
        for i in order:
            source = terma[i] * w
            dose[i] += np.sum(flow * g + source * (1 - g))
            flow = flow * (1 - one_minus_e) + source * one_minus_e
    return np.interp(depths, z, dose)


def fit_hardening(
    kernel: PhotonCollapsedConeKernel,
    reference_depth_dose,
    m: float,
    dz: float,
    ref_depth: float,
) -> tuple[float, float]:
    """
    Fit the effective attenuation ``mu0 / (1 + beta d)`` of the TERMA.

    The collapsed cone broad field depth dose is matched to the reference depth dose between
    10 and 300 mm (both normalized at ``ref_depth``).

    Parameters
    ----------
    kernel : PhotonCollapsedConeKernel
        The kernel.
    reference_depth_dose : callable
        Reference depth dose as a function of the depth [mm].
    m : float
        Attenuation coefficient of the machine [1/mm], start of the search.
    dz : float
        Depth resolution [mm].
    ref_depth : float
        Normalization depth [mm].

    Returns
    -------
    tuple[float, float]
        ``mu0`` and ``beta`` [1/mm].
    """
    depths = np.append(np.arange(10.0, 300.0 + 1e-9, dz), ref_depth)
    reference = reference_depth_dose(depths)
    reference = reference / reference[-1]

    # Relative squared deviation of the normalized depth dose (last depth = normalization)
    def cost(mu, beta):
        dd = broad_field_depth_dose(kernel, mu, depths, dz, beta)
        dd = dd / dd[-1]
        return float(np.sum(((dd[:-1] - reference[:-1]) / reference[:-1]) ** 2))

    # Optimal mu0 for a given beta, searched within a factor 2 of the machine's attenuation
    def fit_mu0(beta):
        log_mu, c = _fminbnd(lambda lm: cost(m * np.exp(lm), beta), np.log(0.5), np.log(2), 1e-8)
        return m * np.exp(log_mu), c

    # Grid search of beta, refined on the neighboring interval
    beta_grid = np.concatenate([[0.0], np.logspace(-5, -2, 16)])
    cost_grid = [fit_mu0(b)[1] for b in beta_grid]
    i_best = int(np.argmin(cost_grid))
    lo = beta_grid[max(i_best - 1, 0)]
    hi = beta_grid[min(i_best + 1, beta_grid.size - 1)]
    beta, _ = _fminbnd(lambda b: fit_mu0(b)[1], lo, hi, 1e-7)
    mu0, _ = fit_mu0(beta)
    return float(mu0), float(beta)


def transport(
    terma: Array,
    rho: Array,
    voxel_size,
    dirs: np.ndarray,
    weights: np.ndarray,
    att: np.ndarray,
    origin=(0, 0, 0),
) -> Array:
    """
    Collapsed cone convolution transport of the TERMA.

    Port of matRad's ``matRad_collapsedConeTransportBatched``. Every direction traverses
    the volume plane by plane along its dominant axis on a sheared lattice, so that every
    voxel is visited exactly once per direction, and along each lattice line the recursion
    of [1]_ is applied to every exponential kernel component with the radiological step
    ``rho * step length``. The lattice is anchored to a global grid through ``origin``, so
    that sub-volumes of the same grid give consistent results.

    All directions (and kernel components) with the same dominant axis and sign are
    transported together in one plane sweep: their flow is kept aligned to the voxels of
    the current plane and shifted by one voxel whenever the lattice shift of a direction
    changes between two planes. Planes are processed with slicing and broadcasting only
    and voxels the flow has not reached yet are skipped. The arrays may belong to any
    Array API namespace (NumPy, CuPy, PyTorch, array-api-strict); the dose is computed in
    the namespace and on the device of ``terma``, in its floating point precision
    (double for other dtypes).

    Parameters
    ----------
    terma : Array
        3D array of TERMA.
    rho : Array
        3D array of relative electron densities (same shape, namespace and device).
    voxel_size : array_like
        Voxel size along the three array axes [mm].
    dirs : np.ndarray
        ``(3, n_dirs)`` unit transport directions in array axes.
    weights, att : np.ndarray
        ``(n_components, n_dirs)`` kernel weights and attenuation coefficients [1/mm water].
    origin : array_like
        Zero based index of the first voxel of the arrays in the global grid.

    Returns
    -------
    Array
        Deposited dose (same shape as ``terma``).

    References
    ----------
    .. [1] Ahnesjo, Med. Phys. 16, 577 (1989)
    """
    xp = array_api_compat.array_namespace(terma, rho)
    if terma.ndim != 3 or tuple(terma.shape) != tuple(rho.shape):
        raise ValueError("TERMA and density must be 3D arrays of the same shape.")
    dirs = np.asarray(dirs, dtype=np.float64).reshape(3, -1)
    weights = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    att = np.atleast_2d(np.asarray(att, dtype=np.float64))
    if weights.shape != att.shape or weights.shape[1] != dirs.shape[1]:
        raise ValueError(
            "Kernel weights and attenuation coefficients must be n_components x n_dirs arrays."
        )
    voxel_size = np.asarray(voxel_size, dtype=np.float64).ravel()
    origin = np.asarray(origin, dtype=np.int64).ravel()

    device = array_api_compat.device(terma)
    dtype = terma.dtype if terma.dtype in (xp.float32, xp.float64) else xp.float64
    terma = xp.astype(terma, dtype, copy=False)
    rho = xp.astype(rho, dtype, copy=False)
    shape = tuple(int(n) for n in terma.shape)
    dose = xp.zeros(shape, dtype=dtype, device=device)
    if not bool(xp.any(terma != 0)):
        return dose

    # Dominant axis of every direction (the first one on a tie) and the sign along it
    num_dirs = dirs.shape[1]
    dominant = np.argmax(np.abs(dirs) / voxel_size[:, None], axis=0)
    forward = dirs[dominant, np.arange(num_dirs)] > 0

    for k in range(3):
        if not np.any(dominant == k):
            continue
        # Transverse axes first and the dominant axis last, so that planes are slices
        perm = ((k + 1) % 3, (k + 2) % 3, k)
        terma_perm = xp.permute_dims(terma, perm)
        rho_perm = xp.permute_dims(rho, perm)
        # Rows and columns of each plane with TERMA and planes with voxels without density,
        # on the host so that the plane loop needs no device queries
        nonzero = terma_perm != 0
        occupancy = {
            "rows": to_numpy(xp.any(nonzero, axis=1)),
            "cols": to_numpy(xp.any(nonzero, axis=0)),
            "void": to_numpy(xp.any(~(rho_perm > 0), axis=(0, 1))),
        }
        dose_perm = xp.zeros(tuple(shape[a] for a in perm), dtype=dtype, device=device)
        for direction_forward in (True, False):
            sel = (dominant == k) & (forward == direction_forward)
            if np.any(sel):
                dose_perm = _transport_group(
                    dose_perm,
                    terma_perm,
                    rho_perm,
                    occupancy,
                    voxel_size[list(perm)],
                    origin[list(perm)],
                    dirs[list(perm)][:, sel],
                    weights[:, sel],
                    att[:, sel],
                    direction_forward,
                )
        dose = dose + xp.permute_dims(dose_perm, tuple(int(a) for a in np.argsort(perm)))
    return dose


def _column_blocks(signs: np.ndarray, start: int = 0) -> list[tuple[int, int, int]]:
    """Contiguous ``(start, stop, sign)`` ranges of a sorted sign vector, without sign 0."""
    blocks = []
    for sign in (-1, 1):
        cols = np.flatnonzero(signs == sign)
        if cols.size:
            blocks.append((start + int(cols[0]), start + int(cols[-1]) + 1, sign))
    return blocks


def _transport_group(dose, terma, rho, occupancy, s, origin, dirs, weights, att, forward):
    """
    Transport all directions of a group with the same dominant axis and sign.

    The arrays are permuted so that the dominant axis is the last one; ``dirs``, ``s`` and
    ``origin`` are permuted accordingly (transverse axis 1, transverse axis 2, dominant
    axis).

    Parameters
    ----------
    dose : Array
        Permuted dose array, the deposited dose is added to it.
    terma, rho : Array
        Permuted arrays of TERMA and relative electron density.
    occupancy : dict
        Host arrays ``rows`` ``(n1, nk)`` and ``cols`` ``(n2, nk)`` of the rows and
        columns with TERMA per plane, and ``void`` ``(nk,)`` of the planes with voxels
        without density.
    s : np.ndarray
        Voxel size along the permuted axes [mm].
    origin : np.ndarray
        Zero based index of the first voxel in the global grid along the permuted axes.
    dirs : np.ndarray
        ``(3, n_dirs)`` directions of the group along the permuted axes.
    weights, att : np.ndarray
        ``(n_components, n_dirs)`` kernel weights and attenuation coefficients.
    forward : bool
        Whether the directions point along the increasing dominant axis.

    Returns
    -------
    Array
        The dose array with the deposited dose added.
    """
    xp = array_api_compat.array_namespace(dose)
    device = array_api_compat.device(dose)
    dtype = dose.dtype
    n1, n2, nk = (int(n) for n in terma.shape)

    # Lattice shifts per plane and direction, rounded on the global grid. Only their change
    # between planes is used, so the offset by the origin of the transverse axes drops out
    step = s[2] / np.abs(dirs[2])
    sigma1 = dirs[0] * s[2] / (dirs[2] * s[0])
    sigma2 = dirs[1] * s[2] / (dirs[2] * s[1])
    global_plane = (np.arange(nk) + origin[2]).astype(np.float64)[:, None]
    sh1 = np.floor(global_plane * sigma1 + 0.5).astype(np.int64)
    sh2 = np.floor(global_plane * sigma2 + 0.5).astype(np.int64)

    # One column per direction and component with non-zero weight. The columns are sorted
    # by the signs of their shifts, so that the columns shifting the same way along a
    # transverse axis are contiguous and are shifted with slices
    comp, col_dir = np.nonzero(weights != 0)
    if col_dir.size == 0:
        return dose
    # Sign of the shifts in the sweep order, against the shear for a backward sweep
    sweep = 1 if forward else -1
    sign1 = sweep * np.sign(sigma1[col_dir]).astype(np.int64)
    sign2 = sweep * np.sign(sigma2[col_dir]).astype(np.int64)
    order = np.lexsort((sign2, sign1))
    comp, col_dir, sign1, sign2 = comp[order], col_dir[order], sign1[order], sign2[order]
    blocks1 = _column_blocks(sign1)
    blocks2 = []
    for value in (-1, 0, 1):
        cols = np.flatnonzero(sign1 == value)
        if cols.size:
            blocks2 += _column_blocks(sign2[cols], int(cols[0]))
    neg_att_step = -att[comp, col_dir] * step[col_dir]
    has_zero_att = bool(np.any(neg_att_step == 0))
    weights_dev = xp.asarray(weights[comp, col_dir], dtype=dtype, device=device)
    neg_att_step_dev = xp.asarray(neg_att_step, dtype=dtype, device=device)
    sum_weights = float(np.sum(weights[comp, col_dir]))

    # Shift of every column between consecutive planes of the sweep (-1, 0 or 1)
    planes = np.arange(nk) if forward else np.arange(nk - 1, -1, -1)
    d1 = np.zeros((nk, col_dir.size), dtype=np.int64)
    d2 = np.zeros_like(d1)
    d1[1:] = np.diff(sh1[planes][:, col_dir], axis=0)
    d2[1:] = np.diff(sh2[planes][:, col_dir], axis=0)
    if np.any(np.abs(d1) > 1) or np.any(np.abs(d2) > 1):
        raise ValueError("Lattice shifts of more than one voxel per plane.")
    moves1 = xp.asarray(d1 != 0, device=device)
    moves2 = xp.asarray(d2 != 0, device=device)

    # The flow carries a border of one voxel that stays zero, so that shifts move zeros in
    # at the edges of the volume; box is the bounding box [lo1, hi1, lo2, hi2] of the
    # non-zero flow in voxel indices, empty at the start
    flow = xp.zeros((n1 + 2, n2 + 2, col_dir.size), dtype=dtype, device=device)
    lo1, hi1, lo2, hi2 = n1, -1, n2, -1

    for q, plane in enumerate(planes.tolist()):
        if q > 0 and hi1 >= lo1 and hi2 >= lo2:
            # Move the flow to the voxels the lattice lines hit in this plane. The
            # destination covers the box plus one voxel, its source is zero outside the box
            r = slice(max(lo1 - 1, 0) + 1, min(hi1 + 1, n1 - 1) + 2)
            c = slice(max(lo2 - 1, 0) + 1, min(hi2 + 1, n2 - 1) + 2)
            for a, b, sign in blocks1:
                src = (slice(r.start - sign, r.stop - sign), c, slice(a, b))
                flow = _shift_columns(
                    xp, flow, (r, c, slice(a, b)), src, d1[q, a:b], moves1[q, a:b]
                )
            for a, b, sign in blocks2:
                src = (r, slice(c.start - sign, c.stop - sign), slice(a, b))
                flow = _shift_columns(
                    xp, flow, (r, c, slice(a, b)), src, d2[q, a:b], moves2[q, a:b]
                )
            lo1 = max(lo1 + int(d1[q].min()), 0)
            hi1 = min(hi1 + int(d1[q].max()), n1 - 1)
            lo2 = max(lo2 + int(d2[q].min()), 0)
            hi2 = min(hi2 + int(d2[q].max()), n2 - 1)

        rows = np.flatnonzero(occupancy["rows"][:, plane])
        if rows.size:
            cols = np.flatnonzero(occupancy["cols"][:, plane])
            lo1, hi1 = min(lo1, int(rows[0])), max(hi1, int(rows[-1]))
            lo2, hi2 = min(lo2, int(cols[0])), max(hi2, int(cols[-1]))
        if hi1 < lo1 or hi2 < lo2:
            continue

        # Ahnesjo's recursion for all columns, with neg_e = -(1 - exp(-x)) and
        # g = (1 - exp(-x)) / x: the deposit is source + (flow - source) g and the new flow
        # flow + (flow - source) neg_e, with source = TERMA * weight
        i1 = slice(lo1, hi1 + 1)
        i2 = slice(lo2, hi2 + 1)
        box = (slice(lo1 + 1, hi1 + 2), slice(lo2 + 1, hi2 + 2), slice(None))
        terma_plane = terma[i1, i2, plane : plane + 1]
        neg_x = rho[i1, i2, plane : plane + 1] * neg_att_step_dev
        neg_e = xp.expm1(neg_x)
        if bool(occupancy["void"][plane]) or has_zero_att:
            # No attenuation over the step: the flow passes unchanged and deposits itself
            attenuated = neg_x < 0
            g = xp.where(attenuated, neg_e / xp.where(attenuated, neg_x, -1.0), 1.0)
        else:
            g = neg_e / neg_x
        flow_box = flow[box]
        flow_minus_source = flow_box - terma_plane * weights_dev
        dose[i1, i2, plane] = (
            dose[i1, i2, plane]
            + terma_plane[:, :, 0] * sum_weights
            + xp.sum(flow_minus_source * g, axis=-1)
        )
        flow[box] = flow_box + flow_minus_source * neg_e
    return dose


def _shift_columns(xp, flow, dst, src, shifts, moves):
    """
    Shift the flow of a contiguous range of columns that move the same way along one axis.

    Parameters
    ----------
    xp : ArrayNamespace
        Namespace of the flow.
    flow : Array
        Padded flow array.
    dst, src : tuple
        Destination and source slices.
    shifts : np.ndarray
        Host shifts of the columns (0 or the common sign).
    moves : Array
        The same as a boolean device array (shift is non-zero).

    Returns
    -------
    Array
        The flow array with the moving columns shifted.
    """
    moving = shifts != 0
    if not np.any(moving):
        return flow
    if np.all(moving):
        # A copy, since source and destination overlap
        flow[dst] = xp.asarray(flow[src], copy=True)
    else:
        flow[dst] = xp.where(moves, flow[src], flow[dst])
    return flow
