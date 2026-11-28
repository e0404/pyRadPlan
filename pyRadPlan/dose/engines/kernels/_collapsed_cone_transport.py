"""
Collapsed cone convolution transport with Array API operations only.

Port of the plane-sweep formulation of matRad's ``matRad_collapsedConeTransportBatched``; it
runs unchanged on NumPy, CuPy, PyTorch and array-api-strict.

References
----------
.. [1] Ahnesjo, Med. Phys. 16, 577 (1989)
"""

import array_api_compat
import numpy as np

from ....core.xp_utils import to_numpy
from ....core.xp_utils.typing import Array, ArrayNamespace


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


def _transport_group(
    dose: Array,
    terma: Array,
    rho: Array,
    occupancy: dict[str, np.ndarray],
    s: np.ndarray,
    origin: np.ndarray,
    dirs: np.ndarray,
    weights: np.ndarray,
    att: np.ndarray,
    forward: bool,
) -> Array:
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
    # by the signs of their shifts (against the shear for a backward sweep), so that the
    # columns shifting the same way along a transverse axis are slices
    comp, col_dir = np.nonzero(weights != 0)
    if col_dir.size == 0:
        return dose
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


def _shift_columns(
    xp: ArrayNamespace,
    flow: Array,
    dst: tuple[slice, ...],
    src: tuple[slice, ...],
    shifts: np.ndarray,
    moves: Array,
) -> Array:
    """Shift the columns of ``flow[dst]`` with a non-zero host shift from ``flow[src]``."""
    moving = shifts != 0
    if not np.any(moving):
        return flow
    if np.all(moving):
        # A copy, since source and destination overlap
        flow[dst] = xp.asarray(flow[src], copy=True)
    else:
        flow[dst] = xp.where(moves, flow[src], flow[dst])
    return flow
