"""
Numba implementation of the collapsed cone transport (port of ``matRad_collapsedConeMex``).

Only imported when numba is installed. The lines of one direction write disjoint voxels,
so they are processed in parallel; lines that never meet a voxel with TERMA are skipped.
"""

import math

import numba
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _source_bounds(terma):
    """
    Bounding boxes of the TERMA in the planes perpendicular to each axis.

    Parameters
    ----------
    terma : np.ndarray
        3D array of TERMA.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(3, 4, max(shape))`` lo1, hi1, lo2, hi2 per axis k and plane (lo > hi: no TERMA in
        the plane) and the ``(3, 4)`` union of the boxes per axis.
    """
    n = terma.shape
    max_n = max(n[0], n[1], n[2])
    # lo1, hi1, lo2, hi2 per axis k and plane p perpendicular to it
    bounds = np.empty((3, 4, max_n), np.int64)
    union = np.empty((3, 4), np.int64)
    for k in range(3):
        n1 = n[(k + 1) % 3]
        n2 = n[(k + 2) % 3]
        for p in range(max_n):
            bounds[k, 0, p] = n1
            bounds[k, 1, p] = -1
            bounds[k, 2, p] = n2
            bounds[k, 3, p] = -1
        union[k, 0] = n1
        union[k, 1] = -1
        union[k, 2] = n2
        union[k, 3] = -1
    sub = np.empty(3, np.int64)
    for a in range(n[0]):
        sub[0] = a
        for b in range(n[1]):
            sub[1] = b
            for c in range(n[2]):
                if terma[a, b, c] == 0.0:
                    continue
                sub[2] = c
                for k in range(3):
                    p = sub[k]
                    u = sub[(k + 1) % 3]
                    v = sub[(k + 2) % 3]
                    bounds[k, 0, p] = min(bounds[k, 0, p], u)
                    bounds[k, 1, p] = max(bounds[k, 1, p], u)
                    bounds[k, 2, p] = min(bounds[k, 2, p], v)
                    bounds[k, 3, p] = max(bounds[k, 3, p], v)
                    union[k, 0] = min(union[k, 0], u)
                    union[k, 1] = max(union[k, 1], u)
                    union[k, 2] = min(union[k, 2], v)
                    union[k, 3] = max(union[k, 3], v)
    return bounds, union


@njit(cache=True)
def _monotone_range(sh, neg_sh, lo, hi):
    """Planes [first, last] with lo <= sh[p] <= hi for a monotone shift sequence."""
    if sh[0] <= sh[sh.size - 1]:
        first = np.searchsorted(sh, lo, side="left")
        last = np.searchsorted(sh, hi, side="right") - 1
    else:
        first = np.searchsorted(neg_sh, -hi, side="left")
        last = np.searchsorted(neg_sh, -lo, side="right") - 1
    return first, last


@njit(cache=True)
def _shifts(nk, origin_k, origin_t, sigma):
    """
    Shifts of the transport lines per plane along one sheared axis.

    Parameters
    ----------
    nk : int
        Number of planes along the dominant axis.
    origin_k, origin_t : int
        Global grid index of the first voxel along the dominant and the sheared axis.
    sigma : float
        Shear of the direction (voxels along the sheared axis per plane).

    Returns
    -------
    np.ndarray
        Shift of the lines in each plane, rounded on the global grid.
    """
    sh = np.empty(nk, np.int64)
    for p in range(nk):
        sh[p] = np.int64(math.floor((p + origin_k) * sigma + 0.5)) - origin_t
    return sh


@njit(parallel=True, cache=True)
def _transport_direction(terma, rho, dose, s, origin, direction, weights, att, bounds, union):
    """
    Transport the TERMA along one cone direction, in parallel over the lines.

    Parameters
    ----------
    terma, rho : np.ndarray
        3D arrays of TERMA and relative electron density.
    dose : np.ndarray
        3D dose array, the deposited dose is added in place.
    s : np.ndarray
        Voxel size along the array axes [mm].
    origin : np.ndarray
        Zero based index of the first voxel in the global grid.
    direction : np.ndarray
        Unit transport direction in array axes.
    weights, att : np.ndarray
        Kernel weights and attenuation coefficients of the components.
    bounds, union : np.ndarray
        Bounding boxes of the TERMA (see ``_source_bounds``).
    """
    # Dominant axis k of the direction: the lines advance one plane perpendicular to k per
    # step and are sheared along the other axes t1, t2
    n = terma.shape
    k = 0
    best = -1.0
    for ax in range(3):
        r = abs(direction[ax]) / s[ax]
        if r > best:
            best = r
            k = ax
    t1 = (k + 1) % 3
    t2 = (k + 2) % 3
    if union[k, 1] < 0:
        return
    strides = (n[1] * n[2], n[2], 1)
    nk = n[k]
    n1 = n[t1]
    n2 = n[t2]

    step = s[k] / abs(direction[k])
    sigma1 = direction[t1] * s[k] / (direction[k] * s[t1])
    sigma2 = direction[t2] * s[k] / (direction[k] * s[t2])
    # Range of lines that cross the bounding box of the TERMA
    sh1 = _shifts(nk, origin[k], origin[t1], sigma1)
    sh2 = _shifts(nk, origin[k], origin[t2], sigma2)
    neg1 = -sh1
    neg2 = -sh2
    min_line1 = union[k, 0] - sh1.max()
    max_line1 = union[k, 1] - sh1.min()
    min_line2 = union[k, 2] - sh2.max()
    max_line2 = union[k, 3] - sh2.min()
    forward = direction[k] > 0.0
    num_comps = weights.size

    flat_terma = terma.ravel()
    flat_rho = rho.ravel()
    flat_dose = dose.reshape(-1)

    for jj in prange(max_line2 - min_line2 + 1):
        j0 = min_line2 + jj
        flow = np.empty(num_comps)
        # Planes where the line is inside the volume (valid) and inside the TERMA box (cand)
        a2, b2 = _monotone_range(sh2, neg2, -j0, n2 - 1 - j0)
        c2, d2 = _monotone_range(sh2, neg2, union[k, 2] - j0, union[k, 3] - j0)
        for i0 in range(min_line1, max_line1 + 1):
            a1, b1 = _monotone_range(sh1, neg1, -i0, n1 - 1 - i0)
            valid_first = max(a1, a2)
            valid_last = min(b1, b2)
            c1, d1 = _monotone_range(sh1, neg1, union[k, 0] - i0, union[k, 1] - i0)
            cand_first = max(c1, c2, valid_first)
            cand_last = min(d1, d2, valid_last)
            if cand_first > cand_last:
                continue
            # The recursion starts at the first voxel of the line with TERMA, as no energy
            # flows along the line before it
            start = -1
            for q in range(cand_last - cand_first + 1):
                p = cand_first + q if forward else cand_last - q
                i = i0 + sh1[p]
                j = j0 + sh2[p]
                if (
                    bounds[k, 0, p] <= i <= bounds[k, 1, p]
                    and bounds[k, 2, p] <= j <= bounds[k, 3, p]
                ):
                    start = p
                    break
            if start < 0:
                continue
            for c in range(num_comps):
                flow[c] = 0.0
            count = valid_last - start + 1 if forward else start - valid_first + 1
            inc = 1 if forward else -1
            p = start
            for _ in range(count):
                ix = p * strides[k] + (i0 + sh1[p]) * strides[t1] + (j0 + sh2[p]) * strides[t2]
                # Ahnesjo's recursion with the radiological step length through the voxel
                dt = step * flat_rho[ix]
                t = flat_terma[ix]
                d = 0.0
                for c in range(num_comps):
                    w = weights[c]
                    if w == 0.0:
                        continue
                    x = att[c] * dt
                    if x > 0.0:
                        one_minus_e = -math.expm1(-x)
                        g = one_minus_e / x
                    else:
                        one_minus_e = 0.0
                        g = 1.0
                    source = t * w
                    d += flow[c] * g + source * (1.0 - g)
                    flow[c] = flow[c] * (1.0 - one_minus_e) + source * one_minus_e
                flat_dose[ix] += d
                p += inc


def transport(terma, rho, voxel_size, dirs, weights, att, origin, num_threads=0):
    """
    Collapsed cone transport on the CPU, see ``_collapsed_cone.transport``.

    Parameters
    ----------
    terma, rho : np.ndarray
        3D arrays (C order, float64) of TERMA and relative electron density.
    voxel_size : np.ndarray
        Voxel size along the array axes [mm].
    dirs : np.ndarray
        ``(3, n_dirs)`` unit transport directions in array axes.
    weights, att : np.ndarray
        ``(n_components, n_dirs)`` kernel weights and attenuation coefficients.
    origin : np.ndarray
        Zero based index of the first voxel in the global grid.
    num_threads : int
        Number of numba threads, 0 for numba's default.

    Returns
    -------
    np.ndarray
        Deposited dose.
    """
    dose = np.zeros_like(terma)
    bounds, union = _source_bounds(terma)
    previous = numba.get_num_threads()
    if num_threads > 0:
        numba.set_num_threads(min(num_threads, numba.config.NUMBA_NUM_THREADS))
    try:
        for d in range(dirs.shape[1]):
            _transport_direction(
                terma,
                rho,
                dose,
                voxel_size,
                origin,
                np.ascontiguousarray(dirs[:, d]),
                np.ascontiguousarray(weights[:, d]),
                np.ascontiguousarray(att[:, d]),
                bounds,
                union,
            )
    finally:
        numba.set_num_threads(previous)
    return dose
