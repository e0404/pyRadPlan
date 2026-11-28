"""
Independent NumPy reference of the collapsed cone transport, used as test oracle.

Direction by direction, vectorized over the lattice lines of each plane with index gathers
(the formulation of matRad's ``matRad_collapsedConeTransport``).
"""

import numpy as np


def reference_transport(terma, rho, voxel_size, dirs, weights, att, origin=(0, 0, 0)):
    """Collapsed cone transport, see ``_collapsed_cone.transport``."""
    terma = np.ascontiguousarray(terma, dtype=np.float64)
    rho = np.ascontiguousarray(rho, dtype=np.float64)
    dirs = np.asarray(dirs, dtype=np.float64).reshape(3, -1)
    weights = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    att = np.atleast_2d(np.asarray(att, dtype=np.float64))
    voxel_size = np.asarray(voxel_size, dtype=np.float64).ravel()
    origin = np.asarray(origin, dtype=np.int64).ravel()
    dose = np.zeros_like(terma)
    for d in range(dirs.shape[1]):
        _transport_direction(
            dose, terma, rho, voxel_size, origin, dirs[:, d], weights[:, d], att[:, d]
        )
    return dose


def _transport_direction(dose, terma, rho, s, origin, direction, weights, att):
    """
    Transport the TERMA along one cone direction (NumPy, vectorized over the lines).

    Parameters
    ----------
    dose : np.ndarray
        3D dose array, the deposited dose is added in place.
    terma, rho : np.ndarray
        3D arrays of TERMA and relative electron density.
    s : np.ndarray
        Voxel size along the array axes [mm].
    origin : np.ndarray
        Zero based index of the first voxel in the global grid.
    direction : np.ndarray
        Unit transport direction in array axes.
    weights, att : np.ndarray
        Kernel weights and attenuation coefficients of the components.
    """
    # Dominant axis k of the direction: the lines advance one plane perpendicular to k per
    # step and are sheared along the other axes t1, t2
    n = terma.shape
    k = int(np.argmax(np.abs(direction) / s))
    t1 = (k + 1) % 3
    t2 = (k + 2) % 3
    strides = (n[1] * n[2], n[2], 1)
    nk = n[k]

    step = s[k] / abs(direction[k])
    sigma1 = direction[t1] * s[k] / (direction[k] * s[t1])
    sigma2 = direction[t2] * s[k] / (direction[k] * s[t2])

    # Shifts of the lines per plane, rounded on the global grid so that sub-volumes of the
    # same grid share the lattice
    global_plane = np.arange(nk) + origin[k]
    sh1 = np.floor(global_plane * sigma1 + 0.5).astype(np.int64) - origin[t1]
    sh2 = np.floor(global_plane * sigma2 + 0.5).astype(np.int64) - origin[t2]

    # Every line (i0, j0) crossing the volume, with its recursion state per component
    i0, j0 = np.meshgrid(
        np.arange(-sh1.max(), n[t1] - sh1.min()),
        np.arange(-sh2.max(), n[t2] - sh2.min()),
        indexing="ij",
    )
    i0 = i0.ravel()
    j0 = j0.ravel()
    flow = np.zeros((i0.size, weights.size))

    flat_terma = terma.ravel()
    flat_rho = rho.ravel()
    flat_dose = dose.reshape(-1)
    # Planes in the transport direction, all lines advance together
    planes = range(nk) if direction[k] > 0 else range(nk - 1, -1, -1)
    for plane in planes:
        i = i0 + sh1[plane]
        j = j0 + sh2[plane]
        valid = np.flatnonzero((i >= 0) & (i < n[t1]) & (j >= 0) & (j < n[t2]))
        if valid.size == 0:
            continue
        # Radiological step length of the lines through the voxels of the plane
        ix = plane * strides[k] + i[valid] * strides[t1] + j[valid] * strides[t2]
        dt = step * flat_rho[ix]
        t = flat_terma[ix]
        deposit = np.zeros(ix.size)
        for c in range(weights.size):
            if weights[c] == 0:
                continue
            # Ahnesjo's recursion: energy entering the voxel along the line (flow) and
            # released in it (source), deposited and attenuated over the step
            x = att[c] * dt
            pos = x > 0
            one_minus_e = np.zeros_like(x)
            g = np.ones_like(x)
            one_minus_e[pos] = -np.expm1(-x[pos])
            g[pos] = one_minus_e[pos] / x[pos]
            source = t * weights[c]
            p = flow[valid, c]
            deposit += p * g + source * (1 - g)
            flow[valid, c] = p * (1 - one_minus_e) + source * one_minus_e
        flat_dose[ix] += deposit
