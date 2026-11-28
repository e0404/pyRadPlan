"""
Collapsed cone convolution: cone directions, TERMA and depth dose normalization.

Port of the matRad collapsed cone engine helpers. The TERMA uses Array API operations only;
the depth dose fits for the normalization run once on the host with NumPy and SciPy. The
transport is in :mod:`.kernels._collapsed_cone_transport`.

References
----------
.. [1] Ahnesjo, Med. Phys. 16, 577 (1989)
.. [2] Bortfeld, Schlegel, Rhein, Med. Phys. 20, 311 (1993)
"""

import array_api_compat
import numpy as np

from pyRadPlan.core.xp_utils import erf, interp1d, scatter
from pyRadPlan.core.xp_utils.typing import Array
from pyRadPlan.machines import PhotonCollapsedConeKernel
from pyRadPlan.machines.photons._collapsed_cone_fit import _fminbnd


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
