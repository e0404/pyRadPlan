"""Array API image interpolation with nearest boundary extrapolation."""

import array_api_compat


def interpolate_image_linear(values, coordinates, *, extrapolate="nearest"):
    """Sample a 3D (z, y, x) image at continuous (x, y, z) indices.

    Match image interpolation semantics: use nearest-pixel extrapolation outside
    the half-voxel image extent, and edge replication within it. Zero-weight
    neighbors do not contribute NaNs. Accumulate in the coordinate precision,
    then return the image dtype. Singleton axes are supported.
    """
    xp = array_api_compat.array_namespace(values, coordinates)
    flat = xp.reshape(values, (-1,))
    nz, ny, nx = values.shape
    limits = (nx, ny, nz)
    lo, hi, weight, nearest = [], [], [], []
    outside = xp.zeros(coordinates.shape[0], dtype=xp.bool, device=array_api_compat.device(values))
    for axis, size in enumerate(limits):
        q = coordinates[:, axis]
        outside = outside | (q < -0.5) | (q >= size - 0.5)
        q = xp.clip(q, 0.0, float(size - 1))
        lower = xp.astype(xp.floor(q), xp.int64)
        upper = xp.minimum(lower + 1, size - 1)
        lo.append(lower)
        hi.append(upper)
        weight.append(q - xp.astype(lower, q.dtype))
        nearest.append(xp.astype(xp.floor(q + 0.5), xp.int64))

    def gather(x, y, z):
        return xp.astype(xp.take(flat, (z * ny + y) * nx + x, axis=0), coordinates.dtype)

    def blend(a, b, t):
        # Do not let an unused NaN neighbor invalidate an exact pixel sample.
        return xp.where(t == 0, a, a + (b - a) * t)

    planes = []
    for z in (lo[2], hi[2]):
        rows = [blend(gather(lo[0], y, z), gather(hi[0], y, z), weight[0]) for y in (lo[1], hi[1])]
        planes.append(blend(rows[0], rows[1], weight[1]))
    linear = blend(planes[0], planes[1], weight[2])
    boundary = (
        gather(*nearest)
        if extrapolate == "nearest"
        else xp.asarray(
            extrapolate, dtype=coordinates.dtype, device=array_api_compat.device(values)
        )
    )
    result = xp.where(outside, boundary, linear)
    return xp.astype(result, values.dtype)


def interpolate_image_nearest(values, coordinates, *, extrapolate="nearest"):
    """Sample nearest voxels without converting integer labels to floating point."""
    xp = array_api_compat.array_namespace(values, coordinates)
    nz, ny, nx = values.shape
    rounded = []
    outside = xp.zeros(coordinates.shape[0], dtype=xp.bool, device=array_api_compat.device(values))
    for axis, size in enumerate((nx, ny, nz)):
        q = coordinates[:, axis]
        outside = outside | (q < -0.5) | (q >= size - 0.5)
        rounded.append(xp.astype(xp.floor(xp.clip(q, 0.0, float(size - 1)) + 0.5), xp.int64))
    x, y, z = rounded
    result = xp.take(xp.reshape(values, (-1,)), (z * ny + y) * nx + x, axis=0)
    if extrapolate != "nearest":
        constant = int(extrapolate) if xp.isdtype(values.dtype, "integral") else extrapolate
        result = xp.where(
            outside,
            xp.asarray(constant, dtype=values.dtype, device=array_api_compat.device(values)),
            result,
        )
    return result
