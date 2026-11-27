"""Numba implementations of the tail sampling kernels for the NumPy backend.

Importing this module registers the implementations on the dispatchable kernels in
:mod:`._tail_sampling_kernel`; it is only imported when numba is installed.
"""

import numpy as np
from numba import njit

from ._tail_sampling_kernel import (
    _ENVELOPE_MARGIN,
    _select_tail_samples,
    _thin_tail_candidates,
)


@njit(cache=True, nogil=True)
def _select_tail_samples_njit(  # noqa: PLR0913, PLR0917 - flat kernel signature
    rad_depths,
    radial_dist_sq,
    numbers,
    envelope,
    grid,
    core_sq,
    slope,
    fac,
    query_shift,
    rotation,
    min_probability,
    depth_min,
    depth_max,
):
    num = rad_depths.shape[0]
    last = grid.shape[0] - 1
    indices = np.empty(num, np.int64)
    weights = np.empty(num, radial_dist_sq.dtype)
    count = 0
    for i in range(num):
        depth = np.float64(rad_depths[i])
        query = depth + query_shift
        if depth < depth_min or depth > depth_max or query < grid[0] or query > grid[last]:
            continue
        low, high = 0, last
        while high - low > 1:
            mid = (low + high) // 2
            if grid[mid] <= query:
                low = mid
            else:
                high = mid
        t = (query - grid[low]) / (grid[high] - grid[low])
        c = core_sq[low] + t * (core_sq[high] - core_sq[low])
        s = slope[low] + t * (slope[high] - slope[low])
        p = np.exp((c - np.float64(radial_dist_sq[i])) * s)
        rotated = np.float64(numbers[i]) + rotation
        if rotated >= 1.0:
            rotated -= 1.0
        env = np.float64(envelope[i])
        if rotated * env < p and p >= min_probability:
            p = min(p, env)
            weight = 1.0
            if p < 1.0:
                weight = (fac[low] + t * (fac[high] - fac[low])) / p
            indices[count] = i
            weights[count] = weight
            count += 1
    return indices[:count].copy(), weights[:count].copy()


@_select_tail_samples.register("numpy")
def _select_tail_samples_numpy(*args):
    if args[4].shape[0] < 2:
        return NotImplemented
    return _select_tail_samples_njit(*args)


@njit(cache=True, nogil=True)
def _thin_tail_candidates_njit(  # noqa: PLR0913, PLR0917 - flat kernel signature
    offsets,
    ix,
    rad_depths,
    radial_dist_sq,
    grid,
    core_sq,
    slope,
    depth_scale,
    depth_shift,
    query_shift,
    rotation,
    min_probability,
):
    num = rad_depths.shape[0]
    last = grid.shape[0] - 1
    keep = np.empty(num, np.int64)
    numbers = np.empty(num, offsets.dtype)
    envelope = np.empty(num, offsets.dtype)
    count = 0
    for i in range(num):
        r_sq = np.float64(radial_dist_sq[i])
        env = 0.0
        for k in range(depth_scale.shape[0]):
            depth = max(depth_scale[k] * np.float64(rad_depths[i]) + depth_shift[k], 0.0)
            query = depth + query_shift
            if query < grid[0] or query > grid[last]:
                continue
            low, high = 0, last
            while high - low > 1:
                mid = (low + high) // 2
                if grid[mid] <= query:
                    low = mid
                else:
                    high = mid
            t = (query - grid[low]) / (grid[high] - grid[low])
            excess = r_sq - (core_sq[low] + t * (core_sq[high] - core_sq[low]))
            p = 1.0
            if excess > 0.0:
                p = np.exp(-excess * (slope[low] + t * (slope[high] - slope[low])))
            env = max(env, p)
        u = np.float64(offsets[ix[i]]) + rotation
        if u >= 1.0:
            u -= 1.0
        if u < env and env >= min_probability:
            keep[count] = i
            numbers[count] = u / env
            envelope[count] = env
            count += 1
    return keep[:count].copy(), numbers[:count].copy(), envelope[:count].copy()


@_thin_tail_candidates.register("numpy")
def _thin_tail_candidates_numpy(*args):
    if args[4].shape[0] < 2:
        return NotImplemented
    return _thin_tail_candidates_njit(*args[:11], _ENVELOPE_MARGIN * args[11])
