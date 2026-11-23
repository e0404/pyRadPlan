"""Resampling with SimpleITK semantics: identity transform, nearest / linear / B-spline."""

from __future__ import annotations

import itertools

import numpy as np

from ._image import Image, cast_array, dtype_of, not_implemented, sitkUnknown

sitkNearestNeighbor = 1
sitkLinear = 2
sitkBSpline1 = 21
sitkBSpline2 = 22
sitkBSpline3 = 23
sitkBSpline4 = 24
sitkBSpline5 = 25
sitkBSpline = sitkBSpline3
sitkGaussian = 4
sitkLabelGaussian = 5
sitkHammingWindowedSinc = 6
sitkCosineWindowedSinc = 7
sitkWelchWindowedSinc = 8
sitkLanczosWindowedSinc = 9
sitkBlackmanWindowedSinc = 10
sitkBSplineResampler = 11
sitkLabelLinear = 26

sitkIdentity = 0

_BSPLINE_ORDER = {
    sitkBSpline1: 1,
    sitkBSpline2: 2,
    sitkBSpline3: 3,
    sitkBSpline4: 4,
    sitkBSpline5: 5,
}

#: Number of output voxels evaluated per chunk on the non-separable paths.
_CHUNK = 1 << 18


class Transform:
    """Identity transform only (the only one pyRadPlan's pipeline resamples with)."""

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], Transform):
            self._dim = args[0]._dim
            return
        self._dim = int(args[0]) if args else 3
        kind = int(args[1]) if len(args) > 1 else sitkIdentity
        if kind != sitkIdentity:
            not_implemented(f"Transform of type {kind} (only identity is supported)")

    def GetDimension(self) -> int:
        return self._dim

    def GetName(self) -> str:
        return "IdentityTransform"

    def TransformPoint(self, point) -> tuple:
        return tuple(float(p) for p in point)

    def IsLinear(self) -> bool:
        return True


def _check_transform(transform) -> None:
    if transform is None or isinstance(transform, Transform):
        return
    not_implemented(f"Resample with transform {type(transform).__name__}")


class ResampleImageFilter:
    """Resample an image onto an output grid (identity transform)."""

    def __init__(self):
        self._size = None
        self._spacing = None
        self._origin = None
        self._direction = None
        self._interpolator = sitkLinear
        self._default = 0.0
        self._pixel_type = sitkUnknown
        self._nn_extrapolator = False
        self._transform = Transform()

    def SetSize(self, size) -> None:
        self._size = tuple(int(s) for s in size)

    def GetSize(self) -> tuple:
        return self._size

    def SetOutputSpacing(self, spacing) -> None:
        self._spacing = tuple(float(s) for s in spacing)

    def GetOutputSpacing(self) -> tuple:
        return self._spacing

    def SetOutputOrigin(self, origin) -> None:
        self._origin = tuple(float(o) for o in origin)

    def GetOutputOrigin(self) -> tuple:
        return self._origin

    def SetOutputDirection(self, direction) -> None:
        self._direction = tuple(float(d) for d in np.asarray(direction).ravel())

    def GetOutputDirection(self) -> tuple:
        return self._direction

    def SetReferenceImage(self, image: Image) -> None:
        self._size = image.GetSize()
        self._spacing = image.GetSpacing()
        self._origin = image.GetOrigin()
        self._direction = image.GetDirection()

    def SetOutputParametersFromImage(self, image: Image) -> None:
        self.SetReferenceImage(image)

    def SetInterpolator(self, interpolator) -> None:
        self._interpolator = int(interpolator)

    def GetInterpolator(self) -> int:
        return self._interpolator

    def SetDefaultPixelValue(self, value) -> None:
        self._default = float(value)

    def GetDefaultPixelValue(self) -> float:
        return self._default

    def SetOutputPixelType(self, pixel_type) -> None:
        self._pixel_type = int(pixel_type)

    def GetOutputPixelType(self) -> int:
        return self._pixel_type

    def SetTransform(self, transform) -> None:
        _check_transform(transform)
        self._transform = transform

    def GetTransform(self):
        return self._transform

    def SetUseNearestNeighborExtrapolator(self, value: bool) -> None:
        self._nn_extrapolator = bool(value)

    def GetUseNearestNeighborExtrapolator(self) -> bool:
        return self._nn_extrapolator

    def UseNearestNeighborExtrapolatorOn(self) -> None:
        self._nn_extrapolator = True

    def UseNearestNeighborExtrapolatorOff(self) -> None:
        self._nn_extrapolator = False

    def Execute(self, image: Image) -> Image:
        dim = image.GetDimension()
        size = self._size if self._size is not None else (0,) * dim
        spacing = self._spacing if self._spacing is not None else (1.0,) * dim
        origin = self._origin if self._origin is not None else (0.0,) * dim
        direction = self._direction if self._direction is not None else tuple(np.eye(dim).ravel())
        if not (len(size) == len(spacing) == len(origin) == dim) or len(direction) != dim * dim:
            raise RuntimeError(f"Output geometry does not match the {dim}D input image.")
        out_dtype = (
            image._array.dtype if self._pixel_type == sitkUnknown else dtype_of(self._pixel_type)
        )
        components = (
            [image._array[..., c] for c in range(image._array.shape[-1])]
            if image._vector
            else [image._array]
        )
        results = [
            _resample_scalar(
                values,
                image,
                size,
                spacing,
                origin,
                direction,
                self._interpolator,
                self._default,
                self._nn_extrapolator,
                out_dtype,
            )
            for values in components
        ]
        array = np.stack(results, axis=-1) if image._vector else results[0]
        out = Image._wrap(array, image._vector)
        out._spacing = tuple(spacing)
        out._origin = tuple(origin)
        out._direction = tuple(direction)
        return out


def Resample(image1: Image, *args, **kwargs) -> Image:  # noqa: PLR0912
    """Resample like ``SimpleITK.Resample`` (reference-image or explicit-geometry form)."""
    names_ref = (
        "referenceImage",
        "transform",
        "interpolator",
        "defaultPixelValue",
        "outputPixelType",
        "useNearestNeighborExtrapolator",
    )
    names_geom = (
        "size",
        "transform",
        "interpolator",
        "outputOrigin",
        "outputSpacing",
        "outputDirection",
        "defaultPixelValue",
        "outputPixelType",
        "useNearestNeighborExtrapolator",
    )
    if args and isinstance(args[0], Image) or "referenceImage" in kwargs:
        params = dict(zip(names_ref, args))
    elif args and isinstance(args[0], (list, tuple)) or "size" in kwargs:
        params = dict(zip(names_geom, args))
    else:
        params = dict(zip(names_ref[1:], args))
        params.setdefault("referenceImage", image1)
    params.update(kwargs)
    f = ResampleImageFilter()
    if "referenceImage" in params:
        f.SetReferenceImage(params["referenceImage"])
    else:
        f.SetSize(params["size"])
        dim = len(params["size"])
        f.SetOutputOrigin(params.get("outputOrigin", (0.0,) * dim))
        f.SetOutputSpacing(params.get("outputSpacing", (1.0,) * dim))
        f.SetOutputDirection(params.get("outputDirection", tuple(np.eye(dim).ravel())))
    f.SetTransform(params.get("transform", Transform()))
    f.SetInterpolator(params.get("interpolator", sitkLinear))
    f.SetDefaultPixelValue(params.get("defaultPixelValue", 0.0))
    f.SetOutputPixelType(params.get("outputPixelType", sitkUnknown))
    f.SetUseNearestNeighborExtrapolator(params.get("useNearestNeighborExtrapolator", False))
    return f.Execute(image1)


def _finalize(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast like ITK's ``CastPixelWithBoundsChecking`` (clamp, then truncate)."""
    if values.dtype == dtype:
        return values
    if np.issubdtype(dtype, np.integer) and not np.issubdtype(values.dtype, np.integer):
        info = np.iinfo(dtype)
        values = np.clip(values, float(info.min), float(info.max))
    return cast_array(values, dtype)


def _default_value(default: float, dtype: np.dtype) -> np.ndarray:
    """Convert the default pixel value with a plain ``static_cast`` (no clamping)."""
    return cast_array(np.asarray(default, dtype=np.float64), dtype)


def _resample_scalar(  # noqa: PLR0913
    values: np.ndarray,
    image: Image,
    size,
    spacing,
    origin,
    direction,
    interpolator: int,
    default: float,
    nn_extrapolator: bool,
    out_dtype: np.dtype,
) -> np.ndarray:
    dim = len(size)
    in_matrix = image._matrix()
    out_matrix = np.reshape(direction, (dim, dim)) * np.asarray(spacing)
    if interpolator not in (sitkNearestNeighbor, sitkLinear) and interpolator not in (
        _BSPLINE_ORDER
    ):
        not_implemented(f"interpolator {interpolator}")
    out_shape = tuple(reversed(size))
    if int(np.prod(size)) == 0:
        return np.zeros(out_shape, dtype=out_dtype)
    in_size = image.GetSize()
    inverse = np.linalg.inv(in_matrix)
    in_origin = np.asarray(image._origin, dtype=np.float64)
    out_origin = np.asarray(origin, dtype=np.float64)

    if _is_diagonal(in_matrix) and _is_diagonal(out_matrix) and interpolator in (1, 2):
        # ITK interpolates the continuous index linearly along each output scanline (x),
        # so x uses that arithmetic and the other axes the direct point transform.
        first = (out_origin[0] - in_origin[0]) * inverse[0, 0]
        last = (out_origin[0] + float(size[0]) * out_matrix[0, 0] - in_origin[0]) * inverse[0, 0]
        fraction = np.arange(size[0], dtype=np.float64) / float(size[0])
        axes = [first + fraction * (last - first)]
        for a in range(1, dim):
            n = np.arange(size[a], dtype=np.float64)
            point = out_origin[a] + n * out_matrix[a, a]
            axes.append((point - in_origin[a]) * inverse[a, a])
        return _separable(values, in_size, axes, interpolator, default, nn_extrapolator, out_dtype)

    def transform(first, rows):
        point = out_origin + first * out_matrix[None, :, 0]
        for a in range(1, dim):
            point = point + rows[a - 1][:, None] * out_matrix[None, :, a]
        delta = point - in_origin
        cindex = delta[:, 0:1] * inverse[None, :, 0]
        for a in range(1, dim):
            cindex = cindex + delta[:, a : a + 1] * inverse[None, :, a]
        return cindex

    row_grid = np.indices(tuple(reversed(size[1:])), dtype=np.float64).reshape(dim - 1, -1)
    rows = [row_grid[dim - 1 - a] for a in range(1, dim)]
    start = transform(0.0, rows)
    vector = transform(float(size[0]), rows) - start
    nx = size[0]
    total = int(np.prod(size))
    out = np.empty(total, dtype=out_dtype)

    source = values
    if interpolator in _BSPLINE_ORDER:
        from scipy import ndimage  # noqa: PLC0415

        order = _BSPLINE_ORDER[interpolator]
        source = ndimage.spline_filter(
            values.astype(np.float64), order=order, mode="mirror", output=np.float64
        )

    for begin in range(0, total, _CHUNK):
        flat = np.arange(begin, min(begin + _CHUNK, total))
        row = flat // nx
        fraction = (flat % nx).astype(np.float64)[:, None] / float(nx)
        q = start[row] + fraction * vector[row]
        out[begin : begin + flat.size] = _evaluate(
            values, source, in_size, q, interpolator, default, nn_extrapolator, out_dtype
        )
    return out.reshape(out_shape)


def _is_diagonal(matrix: np.ndarray) -> bool:
    return bool(np.all(matrix[~np.eye(matrix.shape[0], dtype=bool)] == 0.0))


def _separable(values, in_size, axes, interpolator, default, nn_extrapolator, out_dtype):
    """Evaluate on an axis-aligned output grid, one axis at a time.

    For diagonal index-to-physical matrices the continuous index along an axis depends only on
    the output index along that axis, and blending x, then y, then z on whole planes performs
    exactly the floating point operations of the per-voxel evaluation.
    """
    dim = len(in_size)
    outside = None
    nearest, lower, upper, weight = [], [], [], []
    for a, q in enumerate(axes):
        n = in_size[a]
        out_a = (q < -0.5) | (q >= n - 0.5)
        shape = [1] * dim
        shape[dim - 1 - a] = q.size
        out_a = out_a.reshape(shape)
        outside = out_a if outside is None else (outside | out_a)
        qc = np.clip(q, 0.0, float(n - 1))
        nearest.append(np.floor(qc + 0.5).astype(np.intp))
        lo = np.floor(qc).astype(np.intp)
        lower.append(lo)
        upper.append(np.clip(lo + 1, 0, n - 1))
        weight.append(qc - lo)

    def take_nearest():
        return values[np.ix_(*reversed(nearest))]

    if interpolator == sitkNearestNeighbor:
        result = _finalize(take_nearest(), out_dtype)
        if nn_extrapolator:
            return result
        return np.where(outside, _default_value(default, out_dtype), result)

    current = values.astype(np.float64, copy=False)
    for a in range(dim):
        axis = dim - 1 - a
        lo = np.take(current, lower[a], axis=axis)
        hi = np.take(current, upper[a], axis=axis)
        shape = [1] * dim
        shape[axis] = weight[a].size
        t = weight[a].reshape(shape)
        with np.errstate(invalid="ignore"):
            current = np.where(t == 0, lo, lo + (hi - lo) * t)
    if nn_extrapolator:
        return _finalize(np.where(outside, take_nearest(), current), out_dtype)
    return np.where(outside, _default_value(default, out_dtype), _finalize(current, out_dtype))


def _evaluate(values, source, in_size, q, interpolator, default, nn_extrapolator, out_dtype):
    """Evaluate the interpolator at continuous indices ``q`` (``(n, dim)``, ITK order)."""
    dim = len(in_size)
    outside = np.zeros(q.shape[0], dtype=bool)
    nearest, lower, upper, weight = [], [], [], []
    for a in range(dim):
        n = in_size[a]
        qa = q[:, a]
        outside |= (qa < -0.5) | (qa >= n - 0.5)
        qc = np.clip(qa, 0.0, float(n - 1))
        nearest.append(np.floor(qc + 0.5).astype(np.intp))
        lo = np.floor(qc).astype(np.intp)
        lower.append(lo)
        upper.append(np.clip(lo + 1, 0, n - 1))
        weight.append(qc - lo)

    def gather(idx):
        return values[tuple(reversed(idx))]

    if interpolator == sitkNearestNeighbor:
        result = _finalize(gather(nearest), out_dtype)
        if nn_extrapolator:
            return result
        return np.where(outside, _default_value(default, out_dtype), result)

    if interpolator == sitkLinear:
        corners = {}
        for bits in itertools.product((0, 1), repeat=dim):
            idx = [upper[a] if b else lower[a] for a, b in enumerate(bits)]
            corners[bits] = gather(idx).astype(np.float64)
        for a in range(dim):
            reduced = {}
            for bits, val in corners.items():
                if bits[0] == 1:
                    continue
                hi = corners[(1, *bits[1:])]
                t = weight[a]
                with np.errstate(invalid="ignore"):
                    reduced[bits[1:]] = np.where(t == 0, val, val + (hi - val) * t)
            corners = reduced
        result = corners[()]
    else:
        from scipy import ndimage  # noqa: PLC0415

        coords = np.stack([q[:, a] for a in reversed(range(dim))])
        result = ndimage.map_coordinates(
            source,
            coords,
            order=_BSPLINE_ORDER[interpolator],
            mode="mirror",
            prefilter=False,
        )
    if nn_extrapolator:
        return _finalize(np.where(outside, gather(nearest), result), out_dtype)
    return np.where(outside, _default_value(default, out_dtype), _finalize(result, out_dtype))
