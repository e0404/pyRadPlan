"""Pixel-wise, morphology and label-statistics filters on the numpy-backed image."""

from __future__ import annotations

import math
import numbers

import numpy as np

from ._image import (
    Image,
    cast_array,
    dtype_of,
    not_implemented,
    sitkUInt32,
)

sitkAnnulus = 0
sitkBall = 1
sitkBox = 2
sitkCross = 3

sitkAnyEvent = 0
sitkAbortEvent = 1
sitkDeleteEvent = 2
sitkEndEvent = 3
sitkIterationEvent = 4
sitkProgressEvent = 5
sitkStartEvent = 6
sitkUserEvent = 7


def _same_size(a: Image, b: Image, name: str) -> None:
    if a.GetSize() != b.GetSize():
        raise RuntimeError(f"{name}: inputs do not occupy the same physical space (size).")


def Cast(image: Image, pixelID: int) -> Image:  # noqa: N803
    """Convert the pixel type (C ``static_cast`` semantics)."""
    if image._vector != (12 <= int(pixelID) <= 21):
        not_implemented("Cast between scalar and vector pixel types")
    return Image._wrap(cast_array(image._array, dtype_of(pixelID)), image._vector, like=image)


def _binary_op(image1, image2, func, name, result_dtype=None) -> Image:
    if isinstance(image1, numbers.Number):
        image1, image2 = image2, image1
    if isinstance(image2, Image):
        _same_size(image1, image2, name)
        rhs = image2._array
    else:
        rhs = cast_array(np.asarray(image2), image1._array.dtype)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = func(image1._array, rhs)
    dtype = result_dtype or image1._array.dtype
    return Image._wrap(cast_array(np.asarray(out), dtype), image1._vector, like=image1)


def Add(image1, image2) -> Image:
    return _binary_op(image1, image2, np.add, "Add")


def Subtract(image1, image2) -> Image:
    return image1 - image2


def Multiply(image1, image2) -> Image:
    return _binary_op(image1, image2, np.multiply, "Multiply")


def And(image1, image2) -> Image:
    return _binary_op(image1, image2, np.bitwise_and, "And")


def Or(image1, image2) -> Image:
    return _binary_op(image1, image2, np.bitwise_or, "Or")


def Xor(image1, image2) -> Image:
    return _binary_op(image1, image2, np.bitwise_xor, "Xor")


def Not(image: Image) -> Image:
    return Image._wrap((image._array == 0).astype(image._array.dtype), like=image)


def _compare(image1, image2, func, background, foreground, name) -> Image:
    rhs = image2._array if isinstance(image2, Image) else np.asarray(image2, dtype=np.float64)
    if isinstance(image2, Image):
        _same_size(image1, image2, name)
    hit = func(image1._array, rhs)
    out = np.where(hit, np.uint8(foreground), np.uint8(background)).astype(np.uint8)
    return Image._wrap(out, like=image1)


def Equal(image1, image2, backgroundValue=0, foregroundValue=1) -> Image:  # noqa: N803
    return _compare(image1, image2, np.equal, backgroundValue, foregroundValue, "Equal")


def NotEqual(image1, image2, backgroundValue=0, foregroundValue=1) -> Image:  # noqa: N803
    return _compare(image1, image2, np.not_equal, backgroundValue, foregroundValue, "NotEqual")


def Greater(image1, image2, backgroundValue=0, foregroundValue=1) -> Image:  # noqa: N803
    return _compare(image1, image2, np.greater, backgroundValue, foregroundValue, "Greater")


def Less(image1, image2, backgroundValue=0, foregroundValue=1) -> Image:  # noqa: N803
    return _compare(image1, image2, np.less, backgroundValue, foregroundValue, "Less")


def Mask(image: Image, maskImage: Image, outsideValue=0, maskingValue=0) -> Image:  # noqa: N803
    """Keep ``image`` where ``maskImage != maskingValue``, set ``outsideValue`` elsewhere."""
    _same_size(image, maskImage, "Mask")
    keep = maskImage._array != maskingValue
    if image._vector:
        keep = keep[..., None]
    fill = cast_array(np.asarray(outsideValue), image._array.dtype)
    return Image._wrap(np.where(keep, image._array, fill), image._vector, like=image)


class MaskImageFilter:
    """Object form of :func:`Mask`."""

    def __init__(self):
        self._outside = 0
        self._masking = 0

    def SetOutsideValue(self, value) -> None:
        self._outside = value

    def GetOutsideValue(self):
        return self._outside

    def SetMaskingValue(self, value) -> None:
        self._masking = value

    def GetMaskingValue(self):
        return self._masking

    def Execute(self, image: Image, maskImage: Image) -> Image:  # noqa: N803
        return Mask(image, maskImage, self._outside, self._masking)


def JoinSeries(*images, startingOrigin=0.0, spacing=1.0) -> Image:  # noqa: N803
    """Stack N-D images of equal geometry into an (N+1)-D image."""
    if len(images) == 1 and isinstance(images[0], (list, tuple)):
        images = tuple(images[0])
    if not images:
        raise RuntimeError("JoinSeries needs at least one image.")
    first = images[0]
    for img in images[1:]:
        _same_size(first, img, "JoinSeries")
        if img._array.dtype != first._array.dtype:
            raise RuntimeError("JoinSeries: all inputs must have the same pixel type.")
    array = np.stack([img._array for img in images], axis=0)
    out = Image._wrap(array, first._vector)
    dim = first.GetDimension()
    out._spacing = (*first._spacing, float(spacing))
    out._origin = (*first._origin, float(startingOrigin))
    direction = np.eye(dim + 1)
    direction[:dim, :dim] = np.reshape(first._direction, (dim, dim))
    out._direction = tuple(float(v) for v in direction.ravel())
    return out


def Extract(image: Image, size=None, index=None, directionCollapseToStrategy=None) -> Image:  # noqa: N803
    """Extract a region; zero entries in ``size`` collapse that dimension."""
    dim = image.GetDimension()
    size = list(size) if size is not None else [0] * dim
    index = list(index) if index is not None else [0] * dim
    key = tuple(
        int(i) if int(s) == 0 else slice(int(i), int(i) + int(s)) for s, i in zip(size, index)
    )
    return image[key]


def _structure(radius, dim, kernel_type) -> np.ndarray:
    if isinstance(radius, numbers.Number):
        radius = [int(radius)] * dim
    radius = [int(r) for r in radius]
    shape = tuple(2 * r + 1 for r in reversed(radius))
    if kernel_type == sitkBox:
        return np.ones(shape, dtype=bool)
    if kernel_type == sitkCross:
        structure = np.zeros(shape, dtype=bool)
        center = tuple(r for r in reversed(radius))
        for axis in range(dim):
            sl = list(center)
            sl[axis] = slice(None)
            structure[tuple(sl)] = True
        return structure
    not_implemented(f"structuring element kernel type {kernel_type} (box and cross only)")
    return None


class BinaryDilateImageFilter:
    """Binary dilation with a box or cross structuring element."""

    def __init__(self):
        self._radius = [1]
        self._kernel = sitkBall
        self._foreground = 1.0
        self._background = 0.0
        self._boundary_to_foreground = False

    def SetKernelRadius(self, radius) -> None:
        self._radius = radius

    def GetKernelRadius(self):
        return self._radius

    def SetKernelType(self, kernel_type) -> None:
        self._kernel = int(kernel_type)

    def GetKernelType(self) -> int:
        return self._kernel

    def SetForegroundValue(self, value) -> None:
        self._foreground = float(value)

    def SetBackgroundValue(self, value) -> None:
        self._background = float(value)

    def SetBoundaryToForeground(self, value: bool) -> None:
        self._boundary_to_foreground = bool(value)

    def Execute(self, image: Image) -> Image:
        from scipy import ndimage  # noqa: PLC0415

        radius = self._radius
        if not isinstance(radius, numbers.Number) and len(radius) == 1:
            radius = radius[0]
        dim = image.GetDimension()
        fg = image._array == cast_array(np.asarray(self._foreground), image._array.dtype)
        border = int(self._boundary_to_foreground)
        if self._kernel == sitkBox:
            # A box is separable: one 1D maximum filter per axis.
            radii = [int(radius)] * dim if isinstance(radius, numbers.Number) else radius
            dilated = fg.astype(np.uint8)
            for axis, r in enumerate(reversed([int(r) for r in radii])):
                if r > 0:
                    dilated = ndimage.maximum_filter1d(
                        dilated, size=2 * r + 1, axis=axis, mode="constant", cval=border
                    )
            dilated = dilated.astype(bool)
        else:
            structure = _structure(radius, dim, self._kernel)
            dilated = ndimage.binary_dilation(fg, structure=structure, border_value=border)
        out = image._array.copy()
        out[dilated] = cast_array(np.asarray(self._foreground), out.dtype)
        return Image._wrap(out, like=image)


def BinaryDilate(  # noqa: PLR0913
    image1: Image,
    kernelRadius=(1, 1, 1),  # noqa: N803
    kernelType=sitkBall,  # noqa: N803
    backgroundValue=0.0,  # noqa: N803
    foregroundValue=1.0,  # noqa: N803
    boundaryToForeground=False,  # noqa: N803
) -> Image:
    f = BinaryDilateImageFilter()
    f.SetKernelRadius(kernelRadius)
    f.SetKernelType(kernelType)
    f.SetBackgroundValue(backgroundValue)
    f.SetForegroundValue(foregroundValue)
    f.SetBoundaryToForeground(boundaryToForeground)
    return f.Execute(image1)


def BinaryThreshold(  # noqa: PLR0913
    image1: Image,
    lowerThreshold=0.0,  # noqa: N803
    upperThreshold=255.0,  # noqa: N803
    insideValue=1,  # noqa: N803
    outsideValue=0,  # noqa: N803
) -> Image:
    inside = (image1._array >= lowerThreshold) & (image1._array <= upperThreshold)
    out = np.where(inside, np.uint8(insideValue), np.uint8(outsideValue)).astype(np.uint8)
    return Image._wrap(out, like=image1)


def ConnectedComponent(image: Image, maskImage=None, fullyConnected=False) -> Image:  # noqa: N803
    """Label connected non-zero regions in raster order (``uint32`` labels)."""
    from scipy import ndimage  # noqa: PLC0415

    if isinstance(maskImage, bool):
        fullyConnected, maskImage = maskImage, None
    foreground = image._array != 0
    if maskImage is not None:
        foreground &= maskImage._array != 0
    connectivity = image.GetDimension() if fullyConnected else 1
    structure = ndimage.generate_binary_structure(image.GetDimension(), connectivity)
    labels, _ = ndimage.label(foreground, structure=structure)
    return Image._wrap(labels.astype(dtype_of(sitkUInt32)), like=image)


def BinaryFillhole(image: Image, fullyConnected=False, foregroundValue=1.0) -> Image:  # noqa: N803
    from scipy import ndimage  # noqa: PLC0415

    fg_value = cast_array(np.asarray(foregroundValue), image._array.dtype)
    foreground = image._array == fg_value
    connectivity = image.GetDimension() if fullyConnected else 1
    structure = ndimage.generate_binary_structure(image.GetDimension(), connectivity)
    filled = ndimage.binary_fill_holes(foreground, structure=structure)
    out = image._array.copy()
    out[filled] = fg_value
    return Image._wrap(out, like=image)


class LabelShapeStatisticsImageFilter:
    """Shape statistics per label (background 0), following ITK's ShapeLabelMapFilter.

    Perimeter, Feret diameter and oriented bounding boxes are not implemented.
    """

    def __init__(self):
        self._background = 0
        self._stats = {}

    def SetBackgroundValue(self, value) -> None:
        self._background = value

    def SetComputePerimeter(self, value) -> None:
        if value:
            not_implemented("LabelShapeStatisticsImageFilter perimeter")

    def ComputePerimeterOff(self) -> None:
        pass

    def SetComputeFeretDiameter(self, value) -> None:
        if value:
            not_implemented("LabelShapeStatisticsImageFilter Feret diameter")

    def SetComputeOrientedBoundingBox(self, value) -> None:
        if value:
            not_implemented("LabelShapeStatisticsImageFilter oriented bounding box")

    def Execute(self, image: Image, *_args) -> None:
        if image._vector or not np.issubdtype(image._array.dtype, np.integer):
            raise RuntimeError("LabelShapeStatisticsImageFilter needs an integer label image.")
        array = image._array
        self._stats = {}
        dim = image.GetDimension()
        spacing = np.asarray(image._spacing, dtype=np.float64)
        matrix = image._matrix()
        origin = np.asarray(image._origin, dtype=np.float64)
        labels = np.unique(array)
        for label in labels:
            if label == self._background:
                continue
            idx = np.nonzero(array == label)
            index = np.stack(idx[::-1], axis=1).astype(np.float64)
            count = index.shape[0]
            points = origin + index @ matrix.T
            centroid = points.mean(axis=0)
            centered = points - centroid
            moments = centered.T @ centered / count
            eigvals, eigvecs = np.linalg.eigh(moments)
            axes = eigvecs.T.copy()
            if np.linalg.det(eigvecs) < 0:
                axes[-1] *= -1
            physical_size = count * float(np.prod(spacing))
            radius = _hypersphere_radius(physical_size, dim)
            edet = float(np.prod(eigvals))
            edet = edet ** (1.0 / dim) if edet > 0 else 1.0
            with np.errstate(invalid="ignore", divide="ignore"):
                diameters = 2.0 * radius * np.sqrt(eigvals / edet)
                elongation = math.sqrt(eigvals[-1] / eigvals[-2]) if eigvals[-2] != 0 else 0.0
                flatness = math.sqrt(eigvals[1] / eigvals[0]) if eigvals[0] != 0 else 0.0
            lo = index.min(axis=0).astype(int)
            hi = index.max(axis=0).astype(int)
            self._stats[int(label)] = {
                "count": count,
                "centroid": tuple(float(c) for c in centroid),
                "bbox": tuple(int(v) for v in lo) + tuple(int(v) for v in hi - lo + 1),
                "physical_size": physical_size,
                "moments": tuple(float(v) for v in eigvals),
                "axes": tuple(float(v) for v in axes.ravel()),
                "elongation": elongation,
                "flatness": flatness,
                "diameters": tuple(float(v) for v in diameters),
                "radius": radius,
            }

    def _get(self, label, key):
        label = int(label)
        if label not in self._stats:
            raise RuntimeError(f"No label object with label {label}.")
        return self._stats[label][key]

    def GetLabels(self) -> tuple:
        return tuple(sorted(self._stats))

    def GetNumberOfLabels(self) -> int:
        return len(self._stats)

    def HasLabel(self, label) -> bool:
        return int(label) in self._stats

    def GetNumberOfPixels(self, label) -> int:
        return self._get(label, "count")

    def GetPhysicalSize(self, label) -> float:
        return self._get(label, "physical_size")

    def GetCentroid(self, label) -> tuple:
        return self._get(label, "centroid")

    def GetBoundingBox(self, label) -> tuple:
        return self._get(label, "bbox")

    def GetPrincipalMoments(self, label) -> tuple:
        return self._get(label, "moments")

    def GetPrincipalAxes(self, label) -> tuple:
        return self._get(label, "axes")

    def GetElongation(self, label) -> float:
        return self._get(label, "elongation")

    def GetFlatness(self, label) -> float:
        return self._get(label, "flatness")

    def GetEquivalentEllipsoidDiameter(self, label) -> tuple:
        return self._get(label, "diameters")

    def GetEquivalentSphericalRadius(self, label) -> float:
        return self._get(label, "radius")

    def GetPerimeter(self, label):
        not_implemented("LabelShapeStatisticsImageFilter.GetPerimeter")

    def GetFeretDiameter(self, label):
        not_implemented("LabelShapeStatisticsImageFilter.GetFeretDiameter")

    def GetRoundness(self, label):
        not_implemented("LabelShapeStatisticsImageFilter.GetRoundness")


def _hypersphere_radius(volume: float, dim: int) -> float:
    unit = math.pi ** (dim / 2.0) / math.gamma(dim / 2.0 + 1.0)
    return (volume / unit) ** (1.0 / dim)


def ReadImage(*_args, **_kwargs):
    not_implemented("ReadImage (no image file IO; load .mat data through pyRadPlan.io)")


def WriteImage(*_args, **_kwargs):
    not_implemented("WriteImage (no image file IO)")


class ImageSeriesReader:
    def __init__(self, *_args, **_kwargs):
        not_implemented("ImageSeriesReader (DICOM series import)")


class ImageFileReader:
    def __init__(self, *_args, **_kwargs):
        not_implemented("ImageFileReader")


class ImageFileWriter:
    def __init__(self, *_args, **_kwargs):
        not_implemented("ImageFileWriter")
