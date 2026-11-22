"""numpy-backed ``SimpleITK.Image`` and the array conversion functions."""

from __future__ import annotations

import copy
import numbers
import operator

import numpy as np

sitkUnknown = -1
sitkInt8 = 0
sitkUInt8 = 1
sitkInt16 = 2
sitkUInt16 = 3
sitkInt32 = 4
sitkUInt32 = 5
sitkInt64 = 6
sitkUInt64 = 7
sitkFloat32 = 8
sitkFloat64 = 9
sitkComplexFloat32 = 10
sitkComplexFloat64 = 11
sitkVectorInt8 = 12
sitkVectorUInt8 = 13
sitkVectorInt16 = 14
sitkVectorUInt16 = 15
sitkVectorInt32 = 16
sitkVectorUInt32 = 17
sitkVectorInt64 = 18
sitkVectorUInt64 = 19
sitkVectorFloat32 = 20
sitkVectorFloat64 = 21
sitkLabelUInt8 = 22
sitkLabelUInt16 = 23
sitkLabelUInt32 = 24
sitkLabelUInt64 = 25

_SCALAR_DTYPES = {
    sitkInt8: np.dtype(np.int8),
    sitkUInt8: np.dtype(np.uint8),
    sitkInt16: np.dtype(np.int16),
    sitkUInt16: np.dtype(np.uint16),
    sitkInt32: np.dtype(np.int32),
    sitkUInt32: np.dtype(np.uint32),
    sitkInt64: np.dtype(np.int64),
    sitkUInt64: np.dtype(np.uint64),
    sitkFloat32: np.dtype(np.float32),
    sitkFloat64: np.dtype(np.float64),
    sitkComplexFloat32: np.dtype(np.complex64),
    sitkComplexFloat64: np.dtype(np.complex128),
}
_VECTOR_OFFSET = sitkVectorInt8 - sitkInt8
_DTYPE_IDS = {dtype: pid for pid, dtype in _SCALAR_DTYPES.items()}

_TYPE_NAMES = {
    sitkInt8: "8-bit signed integer",
    sitkUInt8: "8-bit unsigned integer",
    sitkInt16: "16-bit signed integer",
    sitkUInt16: "16-bit unsigned integer",
    sitkInt32: "32-bit signed integer",
    sitkUInt32: "32-bit unsigned integer",
    sitkInt64: "64-bit signed integer",
    sitkUInt64: "64-bit unsigned integer",
    sitkFloat32: "32-bit float",
    sitkFloat64: "64-bit float",
    sitkComplexFloat32: "complex of 32-bit float",
    sitkComplexFloat64: "complex of 64-bit float",
}


def not_implemented(name: str):
    """Raise the shim's uniform error for SimpleITK features it does not provide."""
    raise NotImplementedError(
        f"SimpleITK.{name} is not available in the pyRadPlan Pyodide SimpleITK shim "
        "(web/pyodide/shims/SimpleITK), which implements only the subset pyRadPlan's "
        "planning pipeline uses."
    )


def _is_vector_id(pixel_id: int) -> bool:
    return sitkVectorInt8 <= pixel_id <= sitkVectorFloat64


def dtype_of(pixel_id: int) -> np.dtype:
    """Return the numpy dtype of a (scalar or vector) pixel ID."""
    pixel_id = int(pixel_id)
    if _is_vector_id(pixel_id):
        pixel_id -= _VECTOR_OFFSET
    if pixel_id not in _SCALAR_DTYPES:
        not_implemented(f"pixel type {pixel_id}")
    return _SCALAR_DTYPES[pixel_id]


def pixel_id_of(dtype, vector: bool = False) -> int:
    """Return the pixel ID of a numpy dtype."""
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        raise TypeError("dtype: bool is not supported.")
    if dtype not in _DTYPE_IDS:
        raise TypeError(f"dtype: {dtype} is not supported.")
    pid = _DTYPE_IDS[dtype]
    if vector:
        if pid >= sitkComplexFloat32:
            raise TypeError(f"dtype: {dtype} is not supported for vector images.")
        pid += _VECTOR_OFFSET
    return pid


def cast_array(array: np.ndarray, dtype) -> np.ndarray:
    """Convert like a C ``static_cast``: floats truncate toward zero, integers wrap."""
    dtype = np.dtype(dtype)
    if array.dtype == dtype:
        return array
    if np.issubdtype(dtype, np.integer) and np.issubdtype(array.dtype, np.floating):
        with np.errstate(invalid="ignore", over="ignore"):
            trunc = np.trunc(array)
            if dtype.kind == "u":
                return trunc.astype(np.int64).astype(dtype)
            return trunc.astype(dtype)
    with np.errstate(invalid="ignore", over="ignore"):
        return array.astype(dtype)


def _identity(dim: int) -> tuple:
    return tuple(float(v) for v in np.eye(dim).ravel())


class Image:
    """A numpy-backed stand-in for ``SimpleITK.Image``.

    The voxel buffer is a numpy array in SimpleITK's numpy axis order (``(z, y, x)``, plus a
    trailing component axis for vector images); size, spacing, origin and the index of
    :meth:`GetPixel` are in ITK order (``(x, y, z)``).
    """

    __hash__ = None

    def __init__(self, *args):
        if not args:
            self._init_buffer(np.zeros((0, 0), dtype=np.uint8), vector=False)
            return
        if len(args) == 1 and isinstance(args[0], Image):
            other = args[0]
            self._init_buffer(other._array.copy(), other._vector)
            self.CopyInformation(other)
            self._metadata = dict(other._metadata)
            return
        if isinstance(args[0], (list, tuple)):
            size = [int(s) for s in args[0]]
            pixel_id = int(args[1]) if len(args) > 1 else sitkUInt8
            components = int(args[2]) if len(args) > 2 else 0
        else:
            ints = [int(a) for a in args]
            size, pixel_id, components = ints[:-1], ints[-1], 0
        vector = _is_vector_id(pixel_id)
        shape = tuple(reversed(size))
        if vector:
            shape = (*shape, components if components > 0 else len(size))
        self._init_buffer(np.zeros(shape, dtype=dtype_of(pixel_id)), vector)

    def _init_buffer(self, array: np.ndarray, vector: bool) -> None:
        self._array = array
        self._vector = vector
        dim = array.ndim - (1 if vector else 0)
        self._spacing = (1.0,) * dim
        self._origin = (0.0,) * dim
        self._direction = _identity(dim)
        self._metadata = {}

    @classmethod
    def _wrap(cls, array: np.ndarray, vector: bool = False, like: Image | None = None) -> Image:
        img = cls.__new__(cls)
        img._init_buffer(np.ascontiguousarray(array), vector)
        if like is not None:
            img.CopyInformation(like)
        return img

    def GetDimension(self) -> int:
        return self._array.ndim - (1 if self._vector else 0)

    def GetSize(self) -> tuple:
        shape = self._array.shape[:-1] if self._vector else self._array.shape
        return tuple(int(s) for s in reversed(shape))

    def GetWidth(self) -> int:
        return self.GetSize()[0]

    def GetHeight(self) -> int:
        return self.GetSize()[1]

    def GetDepth(self) -> int:
        size = self.GetSize()
        return size[2] if len(size) > 2 else 0

    def GetNumberOfPixels(self) -> int:
        return int(np.prod(self.GetSize()))

    def GetNumberOfComponentsPerPixel(self) -> int:
        return int(self._array.shape[-1]) if self._vector else 1

    def GetPixelID(self) -> int:
        return pixel_id_of(self._array.dtype, self._vector)

    def GetPixelIDValue(self) -> int:
        return self.GetPixelID()

    def GetPixelIDTypeAsString(self) -> str:
        pid = self.GetPixelID()
        if _is_vector_id(pid):
            return "vector of " + _TYPE_NAMES[pid - _VECTOR_OFFSET]
        return _TYPE_NAMES[pid]

    def GetSpacing(self) -> tuple:
        return self._spacing

    def SetSpacing(self, spacing) -> None:
        spacing = tuple(float(s) for s in spacing)
        self._check_len(spacing, "spacing")
        if any(s <= 0 for s in spacing):
            raise RuntimeError("Spacing must be positive.")
        self._spacing = spacing

    def GetOrigin(self) -> tuple:
        return self._origin

    def SetOrigin(self, origin) -> None:
        origin = tuple(float(o) for o in origin)
        self._check_len(origin, "origin")
        self._origin = origin

    def GetDirection(self) -> tuple:
        return self._direction

    def SetDirection(self, direction) -> None:
        direction = tuple(float(d) for d in np.asarray(direction, dtype=np.float64).ravel())
        dim = self.GetDimension()
        if len(direction) != dim * dim:
            raise RuntimeError(f"Direction must have {dim * dim} elements.")
        if abs(np.linalg.det(np.reshape(direction, (dim, dim)))) < 1e-12:
            raise RuntimeError("Direction matrix is singular.")
        self._direction = direction

    def _check_len(self, values: tuple, name: str) -> None:
        if len(values) != self.GetDimension():
            raise RuntimeError(f"Image {name} must have {self.GetDimension()} elements.")

    def CopyInformation(self, other: Image) -> None:
        if other.GetSize() != self.GetSize():
            raise RuntimeError(
                f"Image sizes differ: {self.GetSize()} vs {other.GetSize()} (CopyInformation)."
            )
        self._spacing = other._spacing
        self._origin = other._origin
        self._direction = other._direction

    def MakeUnique(self) -> None:
        pass

    def IsUnique(self) -> bool:
        return True

    def _matrix(self) -> np.ndarray:
        dim = self.GetDimension()
        return np.reshape(self._direction, (dim, dim)) * np.asarray(self._spacing)

    def TransformContinuousIndexToPhysicalPoint(self, index) -> tuple:
        index = np.asarray(index, dtype=np.float64)
        return tuple(float(v) for v in np.asarray(self._origin) + self._matrix() @ index)

    def TransformIndexToPhysicalPoint(self, index) -> tuple:
        return self.TransformContinuousIndexToPhysicalPoint([int(i) for i in index])

    def TransformPhysicalPointToContinuousIndex(self, point) -> tuple:
        delta = np.asarray(point, dtype=np.float64) - np.asarray(self._origin)
        return tuple(float(v) for v in np.linalg.solve(self._matrix(), delta))

    def TransformPhysicalPointToIndex(self, point) -> tuple:
        cindex = self.TransformPhysicalPointToContinuousIndex(point)
        return tuple(int(np.floor(c + 0.5)) for c in cindex)

    def EvaluateAtContinuousIndex(self, index, interp=None):
        not_implemented("Image.EvaluateAtContinuousIndex")

    def _numpy_index(self, index) -> tuple:
        size = self.GetSize()
        index = [int(i) for i in index]
        if len(index) != len(size):
            raise IndexError(f"Index must have {len(size)} elements.")
        for i, s in zip(index, size):
            if not 0 <= i < s:
                raise IndexError("index out of bounds")
        return tuple(reversed(index))

    def GetPixel(self, *index):
        if len(index) == 1 and isinstance(index[0], (list, tuple)):
            index = index[0]
        value = self._array[self._numpy_index(index)]
        if self._vector:
            return tuple(v.item() for v in value)
        return value.item()

    def SetPixel(self, *args) -> None:
        if len(args) == 2 and isinstance(args[0], (list, tuple)):
            index, value = args
        else:
            index, value = args[:-1], args[-1]
        self._array[self._numpy_index(index)] = value

    def __getitem__(self, key):
        dim = self.GetDimension()
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) != dim:
            raise IndexError(f"Image index must have {dim} entries.")
        if all(isinstance(k, numbers.Integral) for k in key):
            size = self.GetSize()
            return self.GetPixel([int(k) + s if k < 0 else int(k) for k, s in zip(key, size)])
        size = self.GetSize()
        starts, steps, keep, np_key = [], [], [], []
        for k, s in zip(key, size):
            if isinstance(k, numbers.Integral):
                i = int(k) + s if k < 0 else int(k)
                if not 0 <= i < s:
                    raise IndexError("index out of bounds")
                starts.append(i)
                steps.append(1)
                np_key.append(i)
                continue
            if not isinstance(k, slice):
                raise TypeError("Image indices must be integers or slices.")
            start, _, step = k.indices(s)
            if step < 0:
                not_implemented("Image.__getitem__ with negative slice steps")
            starts.append(start)
            steps.append(step)
            keep.append(len(starts) - 1)
            np_key.append(k)
        array = self._array[tuple(reversed(np_key)) + ((slice(None),) if self._vector else ())]
        out = Image._wrap(array.copy(), self._vector)
        origin = self.TransformContinuousIndexToPhysicalPoint(starts)
        direction = np.reshape(self._direction, (dim, dim))
        out._origin = tuple(origin[i] for i in keep)
        out._spacing = tuple(self._spacing[i] * steps[i] for i in keep)
        sub = direction[np.ix_(keep, keep)]
        if abs(np.linalg.det(sub)) < 1e-12:
            sub = np.eye(len(keep))
        out._direction = tuple(float(v) for v in sub.ravel())
        return out

    def __setitem__(self, key, value):
        if not isinstance(key, tuple) or not all(isinstance(k, numbers.Integral) for k in key):
            not_implemented("Image.__setitem__ with slices")
        self.SetPixel(list(key), value)

    def __copy__(self):
        return Image(self)

    def __deepcopy__(self, memo):
        return Image(self)

    def GetMetaDataKeys(self) -> tuple:
        return tuple(self._metadata)

    def HasMetaDataKey(self, key: str) -> bool:
        return key in self._metadata

    def GetMetaData(self, key: str) -> str:
        if key not in self._metadata:
            raise RuntimeError(f"Key '{key}' does not exist in the meta-data dictionary.")
        return self._metadata[key]

    def SetMetaData(self, key: str, value: str) -> None:
        self._metadata[str(key)] = str(value)

    def EraseMetaData(self, key: str) -> bool:
        return self._metadata.pop(key, None) is not None

    def __reduce_ex__(self, protocol):
        # Same reduce tuple as SimpleITK 2.x (``SimpleITK.SimpleITK.Image``, state version 0),
        # so pickles move between this shim and real SimpleITK in both directions.
        buffer = np.ascontiguousarray(self._array)
        if protocol >= 5:
            import pickle  # noqa: PLC0415

            data = pickle.PickleBuffer(buffer)
        else:
            data = buffer.tobytes()
        state = (0, data, self._origin, self._spacing, self._direction, dict(self._metadata))
        args = (self.GetSize(), self.GetPixelIDValue(), self.GetNumberOfComponentsPerPixel())
        return self.__class__, args, state

    def __setstate__(self, state):
        if state[0] != 0:
            raise ValueError(f"Unable to handle SimpleITK.Image pickle version {state[0]}")
        _, data, origin, spacing, direction, metadata = state
        values = np.frombuffer(data, dtype=self._array.dtype)
        self._array = values.reshape(self._array.shape).copy()
        self.SetOrigin(origin)
        self.SetSpacing(spacing)
        self.SetDirection(direction)
        self._metadata = {str(k): str(v) for k, v in metadata.items()}

    def __repr__(self):
        return (
            f"<SimpleITK shim Image {self.GetSize()} {self.GetPixelIDTypeAsString()} "
            f"spacing={self._spacing} origin={self._origin}>"
        )

    def __iter__(self):
        not_implemented("Image.__iter__")

    def _binary(self, other, op, result_dtype=None):
        if isinstance(other, Image):
            if other.GetSize() != self.GetSize():
                raise RuntimeError("Images must have the same size.")
            if other._array.dtype != self._array.dtype:
                raise RuntimeError(
                    "Exception thrown in SimpleITK shim: both images must have the same pixel "
                    f"type ({self.GetPixelIDTypeAsString()} vs {other.GetPixelIDTypeAsString()})."
                )
            rhs = other._array
        elif isinstance(other, numbers.Number):
            rhs = cast_array(np.asarray(other), self._array.dtype)
        else:
            return NotImplemented
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            out = op(self._array, rhs)
        out = cast_array(np.asarray(out), result_dtype or self._array.dtype)
        return Image._wrap(out, self._vector, like=self)

    def _rbinary(self, other, op):
        if not isinstance(other, numbers.Number):
            return NotImplemented
        lhs = cast_array(np.asarray(other), self._array.dtype)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            out = op(lhs, self._array)
        return Image._wrap(cast_array(out, self._array.dtype), self._vector, like=self)

    def __add__(self, other):
        return self._binary(other, operator.add)

    def __radd__(self, other):
        return self._rbinary(other, operator.add)

    def __sub__(self, other):
        return self._binary(other, operator.sub)

    def __rsub__(self, other):
        return self._rbinary(other, operator.sub)

    def __mul__(self, other):
        return self._binary(other, operator.mul)

    def __rmul__(self, other):
        return self._rbinary(other, operator.mul)

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            other = float(other)
            out = np.true_divide(self._array.astype(np.float64), other)
            return Image._wrap(out, self._vector, like=self)
        if isinstance(other, Image):
            return self._binary(
                other, lambda a, b: np.true_divide(a.astype(np.float64), b), np.float64
            )
        return NotImplemented

    def __floordiv__(self, other):
        return self._binary(other, np.floor_divide)

    def __mod__(self, other):
        return self._binary(other, np.mod)

    def __pow__(self, other):
        return self._binary(other, np.power)

    def __neg__(self):
        return Image._wrap(-self._array, self._vector, like=self)

    def __pos__(self):
        return Image(self)

    def __abs__(self):
        return Image._wrap(np.abs(self._array), self._vector, like=self)

    def _require_integer(self, name):
        if not np.issubdtype(self._array.dtype, np.integer):
            raise RuntimeError(f"{name} requires an integer pixel type.")

    def __and__(self, other):
        self._require_integer("And")
        return self._binary(other, np.bitwise_and)

    def __or__(self, other):
        self._require_integer("Or")
        return self._binary(other, np.bitwise_or)

    def __xor__(self, other):
        self._require_integer("Xor")
        return self._binary(other, np.bitwise_xor)

    def __invert__(self):
        self._require_integer("Not")
        return Image._wrap(np.invert(self._array), self._vector, like=self)

    __rand__ = __and__
    __ror__ = __or__
    __rxor__ = __xor__

    def _compare(self, other, op):
        if not isinstance(other, (Image, numbers.Number)):
            return NotImplemented
        if isinstance(other, numbers.Number):
            other = np.asarray(other, dtype=np.float64)
            out = op(self._array.astype(np.float64, copy=False), other)
        else:
            if other.GetSize() != self.GetSize():
                raise RuntimeError("Images must have the same size.")
            out = op(self._array, other._array)
        return Image._wrap(out.astype(np.uint8), self._vector, like=self)

    def __eq__(self, other):
        return self._compare(other, operator.eq)

    def __ne__(self, other):
        return self._compare(other, operator.ne)

    def __lt__(self, other):
        return self._compare(other, operator.lt)

    def __le__(self, other):
        return self._compare(other, operator.le)

    def __gt__(self, other):
        return self._compare(other, operator.gt)

    def __ge__(self, other):
        return self._compare(other, operator.ge)


# Pickles reference the class where real SimpleITK defines it (``SimpleITK.py`` here).
Image.__module__ = "SimpleITK.SimpleITK"


def GetArrayFromImage(image: Image) -> np.ndarray:
    """Return a copy of the voxel buffer in numpy axis order."""
    return image._array.copy()


def GetArrayViewFromImage(image: Image) -> np.ndarray:
    """Return a read-only view of the voxel buffer in numpy axis order."""
    view = image._array.view()
    view.flags.writeable = False
    return view


def GetImageFromArray(arr, isVector: bool = False) -> Image:  # noqa: N803
    """Copy a numpy array (``(z, y, x)`` order) into a new image."""
    array = np.array(arr, copy=True, order="C")
    if array.dtype == np.bool_:
        raise TypeError("dtype: bool is not supported.")
    vector = bool(isVector) and array.ndim >= 3
    if array.ndim - (1 if vector else 0) < 2 or array.ndim > 5:
        raise RuntimeError(f"Unsupported number of dimensions: {array.ndim}.")
    if array.dtype.byteorder not in ("=", "|"):
        array = array.astype(array.dtype.newbyteorder("="))
    pixel_id_of(array.dtype, vector)
    return Image._wrap(array, vector)


def deepcopy_image(image: Image) -> Image:
    """Return an independent copy (used by filters that modify their input buffer)."""
    return copy.deepcopy(image)
