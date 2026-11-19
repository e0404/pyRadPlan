"""Compare the SimpleITK shim against real SimpleITK."""

import itertools

import numpy as np
import pytest


def pair(sitk, shim, array, spacing=None, origin=None, direction=None, vector=False):
    images = []
    for mod in (sitk, shim):
        img = mod.GetImageFromArray(array, isVector=vector)
        if spacing is not None:
            img.SetSpacing(spacing)
        if origin is not None:
            img.SetOrigin(origin)
        if direction is not None:
            img.SetDirection(direction)
        images.append(img)
    return images


def assert_same_geometry(a, b):
    assert a.GetSize() == b.GetSize()
    assert a.GetDimension() == b.GetDimension()
    np.testing.assert_allclose(a.GetSpacing(), b.GetSpacing(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(a.GetOrigin(), b.GetOrigin(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(a.GetDirection(), b.GetDirection(), rtol=0, atol=1e-12)
    assert a.GetNumberOfComponentsPerPixel() == b.GetNumberOfComponentsPerPixel()


def assert_same(sitk, shim, a, b, rtol=0.0, atol=0.0):
    assert_same_geometry(a, b)
    assert a.GetPixelID() == b.GetPixelID()
    assert a.GetPixelIDTypeAsString() == b.GetPixelIDTypeAsString()
    x, y = sitk.GetArrayFromImage(a), shim.GetArrayFromImage(b)
    assert x.dtype == y.dtype
    np.testing.assert_allclose(y, x, rtol=rtol, atol=atol)


def rotation(seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return tuple(q.ravel())


DTYPES = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]
DTYPES += [np.float32, np.float64]


@pytest.mark.parametrize("dtype", DTYPES)
def test_pixel_types_and_roundtrip(sitk, shim, dtype):
    arr = (np.arange(60) % 7).reshape(3, 4, 5).astype(dtype)
    a, b = pair(sitk, shim, arr)
    assert_same(sitk, shim, a, b)
    assert a.GetSize() == b.GetSize() == (5, 4, 3)
    view = shim.GetArrayViewFromImage(b)
    assert not view.flags.writeable
    assert a.GetPixel(4, 3, 2) == b.GetPixel(4, 3, 2)
    assert a[1, 2, 0] == b[1, 2, 0]


def test_bool_rejected(sitk, shim):
    for mod in (sitk, shim):
        with pytest.raises(TypeError):
            mod.GetImageFromArray(np.ones((2, 2, 2), dtype=bool))


def test_vector_and_constructor(sitk, shim):
    arr = np.random.default_rng(0).random((3, 4, 5, 2)).astype(np.float32)
    a, b = pair(sitk, shim, arr, vector=True)
    assert_same(sitk, shim, a, b)
    for args in ([(4, 3, 2), 8], [(4, 3, 2), 1], [(4, 3), 9], [(4, 3, 2), 20, 3]):
        assert_same(sitk, shim, sitk.Image(*args), shim.Image(*args))
    assert_same(sitk, shim, sitk.Image(4, 3, 2, 8), shim.Image(4, 3, 2, 8))
    assert sitk.Image().GetSize() == shim.Image().GetSize()
    assert sitk.Image().GetPixelIDValue() == shim.Image().GetPixelIDValue()


def test_constants_match(sitk, shim):
    names = [n for n in dir(shim) if n.startswith("sitk")]
    assert len(names) > 30
    for name in names:
        assert getattr(sitk, name) == getattr(shim, name), name


def test_physical_point_transforms(sitk, shim):
    arr = np.zeros((6, 7, 8), dtype=np.float32)
    a, b = pair(sitk, shim, arr, (1.5, 2.0, 3.25), (-10.0, 5.5, 2.0), rotation(1))
    for idx in [(0, 0, 0), (3, 2, 5), (7, 6, 5)]:
        np.testing.assert_allclose(
            b.TransformIndexToPhysicalPoint(idx), a.TransformIndexToPhysicalPoint(idx), atol=1e-12
        )
    cidx = (1.3, -0.4, 4.7)
    p = a.TransformContinuousIndexToPhysicalPoint(cidx)
    np.testing.assert_allclose(b.TransformContinuousIndexToPhysicalPoint(cidx), p, atol=1e-12)
    np.testing.assert_allclose(
        b.TransformPhysicalPointToContinuousIndex(p),
        a.TransformPhysicalPointToContinuousIndex(p),
        atol=1e-10,
    )
    assert b.TransformPhysicalPointToIndex(p) == a.TransformPhysicalPointToIndex(p)


def test_slicing_and_join(sitk, shim):
    arr = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    a, b = pair(sitk, shim, arr, (2.0, 3.0, 4.0), (1.0, 2.0, 3.0))
    for key in [
        (slice(1, 3), slice(None), 1),
        (slice(None, None, 2), slice(None), slice(None)),
        (slice(None), 1, slice(None)),
        (slice(1, None), slice(0, 2), slice(1, 2)),
    ]:
        assert_same(sitk, shim, a[key], b[key])
    ja, jb = sitk.JoinSeries([a, a]), shim.JoinSeries([b, b])
    assert_same(sitk, shim, ja, jb)
    assert_same(sitk, shim, sitk.JoinSeries(a, a, a), shim.JoinSeries(b, b, b))
    assert_same(sitk, shim, ja[:, :, :, 1], jb[:, :, :, 1])
    ea = sitk.Extract(a, [4, 3, 0], [0, 0, 1])
    eb = shim.Extract(b, [4, 3, 0], [0, 0, 1])
    assert_same(sitk, shim, ea, eb)
    ca, cb = sitk.Image(a), shim.Image(b)
    assert_same(sitk, shim, ca, cb)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32, np.float64])
def test_arithmetic_and_comparisons(sitk, shim, dtype):
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 20, size=(3, 4, 5)).astype(dtype)
    arr2 = rng.integers(1, 20, size=(3, 4, 5)).astype(dtype)
    a, b = pair(sitk, shim, arr, (1.0, 2.0, 3.0))
    a2, b2 = pair(sitk, shim, arr2, (1.0, 2.0, 3.0))
    ops = [
        lambda x, y: x + y,
        lambda x, y: x - y,
        lambda x, y: x * y,
        lambda x, y: x / y,
        lambda x, y: x // y,
        lambda x, _: x + 3,
        lambda x, _: x - 25,
        lambda x, _: x * 1.5,
        lambda x, _: x / 3,
        lambda x, _: 2 * x,
        lambda x, _: x > 5,
        lambda x, _: x != 0,
        lambda x, y: x == y,
        lambda x, _: x <= 7.5,
        lambda x, _: -x,
    ]
    for op in ops:
        try:
            expected = op(a, a2)
        except RuntimeError:  # pixel type not supported by that SimpleITK filter
            continue
        assert_same(sitk, shim, expected, op(b, b2), rtol=1e-12)
    for cast in (sitk.sitkUInt8, sitk.sitkInt16, sitk.sitkFloat32, sitk.sitkFloat64):
        assert_same(sitk, shim, sitk.Cast(a * 7.3, cast), shim.Cast(b * 7.3, cast))
    assert (a == None) is False  # noqa: E711
    assert (b == None) is False  # noqa: E711
    with pytest.raises(TypeError):
        hash(b)


def test_cast_float_to_int(sitk, shim):
    arr = np.array([[[-1.7, 1.7, 2.5, 300.2, -0.2]]], dtype=np.float32)
    a, b = pair(sitk, shim, arr)
    for cast in (sitk.sitkUInt8, sitk.sitkInt16, sitk.sitkInt8, sitk.sitkUInt16):
        assert_same(sitk, shim, sitk.Cast(a, cast), shim.Cast(b, cast))


def test_mask_and_logic(sitk, shim):
    rng = np.random.default_rng(4)
    img = rng.normal(size=(4, 5, 6)).astype(np.float32)
    mask = rng.integers(0, 2, size=(4, 5, 6)).astype(np.uint8)
    mask2 = rng.integers(0, 2, size=(4, 5, 6)).astype(np.uint8)
    a, b = pair(sitk, shim, img, (2.0, 2.0, 3.0))
    ma, mb = pair(sitk, shim, mask, (2.0, 2.0, 3.0))
    m2a, m2b = pair(sitk, shim, mask2, (2.0, 2.0, 3.0))
    assert_same(sitk, shim, sitk.Mask(a, ma), shim.Mask(b, mb))
    assert_same(
        sitk, shim, sitk.Mask(a, ma, outsideValue=-5.0), shim.Mask(b, mb, outsideValue=-5.0)
    )
    fa, fb = sitk.MaskImageFilter(), shim.MaskImageFilter()
    assert_same(sitk, shim, fa.Execute(a, ma), fb.Execute(b, mb))
    assert_same(sitk, shim, sitk.And(ma, m2a), shim.And(mb, m2b))
    assert_same(sitk, shim, sitk.Or(ma, m2a), shim.Or(mb, m2b))
    assert_same(sitk, shim, sitk.Equal(ma, 0), shim.Equal(mb, 0))
    assert_same(sitk, shim, sitk.Add(ma, m2a), shim.Add(mb, m2b))
    assert_same(sitk, shim, sitk.Multiply(ma, m2a), shim.Multiply(mb, m2b))
    assert_same(sitk, shim, sitk.Multiply(ma, m2a) * 3, shim.Multiply(mb, m2b) * 3)


def test_morphology(sitk, shim):
    rng = np.random.default_rng(5)
    mask = (rng.random((12, 14, 16)) > 0.93).astype(np.uint8)
    ma, mb = pair(sitk, shim, mask)
    for radius in ([1, 2, 3], [2, 0, 1], [1, 1, 1], [6, 5, 4]):
        da, db = sitk.BinaryDilateImageFilter(), shim.BinaryDilateImageFilter()
        for f in (da, db):
            f.SetKernelType(sitk.sitkBox)
            f.SetKernelRadius(radius)
        assert_same(sitk, shim, da.Execute(ma), db.Execute(mb))
    ct = rng.normal(size=(12, 14, 16)).astype(np.float32)
    ca, cb = pair(sitk, shim, ct)
    assert_same(
        sitk,
        shim,
        sitk.BinaryThreshold(ca, -0.5, 0.7, 1, 0),
        shim.BinaryThreshold(cb, -0.5, 0.7, 1, 0),
    )
    blobs = (rng.random((12, 14, 16)) > 0.7).astype(np.uint8)
    ba, bb = pair(sitk, shim, blobs)
    assert_same(sitk, shim, sitk.ConnectedComponent(ba), shim.ConnectedComponent(bb))
    ring = np.zeros((9, 9), dtype=np.uint8)
    ring[2:7, 2:7] = 1
    ring[4, 4] = 0
    ra, rb = pair(sitk, shim, ring)
    assert_same(sitk, shim, sitk.BinaryFillhole(ra), shim.BinaryFillhole(rb))


@pytest.mark.parametrize("direction_seed", [None, 7])
def test_label_shape_statistics(sitk, shim, direction_seed):
    zz, yy, xx = np.mgrid[0:20, 0:24, 0:28]
    ellipsoid = ((xx - 13.2) / 9) ** 2 + ((yy - 10.7) / 5) ** 2 + ((zz - 9.1) / 3) ** 2 <= 1
    ellipsoid |= (xx > 20) & (yy > 18) & (zz < 3)
    labels = ellipsoid.astype(np.uint8)
    labels[0:2, 0:3, 0:4] = 2
    direction = rotation(direction_seed) if direction_seed else None
    a, b = pair(sitk, shim, labels, (1.5, 2.0, 2.5), (3.0, -4.0, 1.0), direction)
    sa, sb = sitk.LabelShapeStatisticsImageFilter(), shim.LabelShapeStatisticsImageFilter()
    sa.Execute(a)
    sb.Execute(b)
    assert sa.GetLabels() == sb.GetLabels()
    assert sb.HasLabel(1) and not sb.HasLabel(3)
    for label in sa.GetLabels():
        assert sa.GetNumberOfPixels(label) == sb.GetNumberOfPixels(label)
        assert sa.GetBoundingBox(label) == sb.GetBoundingBox(label)
        for getter in (
            "GetPhysicalSize",
            "GetCentroid",
            "GetPrincipalMoments",
            "GetElongation",
            "GetFlatness",
            "GetEquivalentEllipsoidDiameter",
            "GetEquivalentSphericalRadius",
        ):
            np.testing.assert_allclose(
                getattr(sb, getter)(label), getattr(sa, getter)(label), rtol=1e-9, err_msg=getter
            )
        axes_a = np.reshape(sa.GetPrincipalAxes(label), (3, 3))
        axes_b = np.reshape(sb.GetPrincipalAxes(label), (3, 3))
        np.testing.assert_allclose(np.abs(np.sum(axes_a * axes_b, axis=1)), 1.0, atol=1e-9)
        np.testing.assert_allclose(np.linalg.det(axes_b), np.linalg.det(axes_a), atol=1e-9)


def _resample_both(sitk, shim, a, b, geometry, interpolator, default, nn, pixel_type):
    outs = []
    for mod, img in ((sitk, a), (shim, b)):
        f = mod.ResampleImageFilter()
        size, spacing, origin, direction = geometry
        f.SetSize(size)
        f.SetOutputSpacing(spacing)
        f.SetOutputOrigin(origin)
        f.SetOutputDirection(direction)
        f.SetInterpolator(interpolator)
        f.SetDefaultPixelValue(default)
        f.SetUseNearestNeighborExtrapolator(nn)
        if pixel_type is not None:
            f.SetOutputPixelType(pixel_type)
        outs.append(f.Execute(img))
    return outs


GEOMETRIES = {
    # TG119-like: 3 mm CT resampled to a 5 mm dose grid and back (aligned voxel boundaries)
    "ct_to_dose": (
        ((40, 38, 30), (3.0, 3.0, 2.5), (0.0, 0.0, 0.0), None),
        ((25, 23, 16), (5.0, 5.0, 5.0), (1.0, 1.0, 1.25), None),
    ),
    "dose_to_ct": (
        ((25, 23, 16), (5.0, 5.0, 5.0), (1.0, 1.0, 1.25), None),
        ((40, 38, 30), (3.0, 3.0, 2.5), (0.0, 0.0, 0.0), None),
    ),
    "same_grid": (
        ((20, 21, 22), (1.7, 2.3, 3.1), (-5.0, 3.0, 7.0), None),
        ((20, 21, 22), (1.7, 2.3, 3.1), (-5.0, 3.0, 7.0), None),
    ),
    "shifted_partial": (
        ((20, 21, 22), (2.0, 2.0, 2.0), (0.0, 0.0, 0.0), None),
        ((30, 9, 25), (1.3, 3.7, 1.9), (-7.0, 11.0, -3.3), None),
    ),
    "rotated_input": (
        ((18, 19, 20), (2.0, 2.5, 3.0), (0.0, 0.0, 0.0), 11),
        ((22, 22, 22), (2.1, 2.1, 2.1), (-15.0, -10.0, -5.0), None),
    ),
    "rotated_both": (
        ((18, 19, 20), (2.0, 2.5, 3.0), (0.0, 0.0, 0.0), 11),
        ((15, 16, 17), (2.6, 2.4, 2.2), (-6.0, 2.0, 1.0), 12),
    ),
}


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
@pytest.mark.parametrize(
    "interpolator,dtype",
    [
        ("nearest", np.uint8),
        ("nearest", np.int64),
        ("nearest", np.float32),
        ("linear", np.float64),
        ("linear", np.float32),
        ("linear", np.uint8),
        ("bspline", np.float64),
    ],
)
@pytest.mark.parametrize("nn", [False, True])
def test_resample(sitk, shim, name, interpolator, dtype, nn):
    (in_size, in_sp, in_or, in_dir), (out_size, out_sp, out_or, out_dir) = GEOMETRIES[name]
    rng = np.random.default_rng(abs(hash((name, interpolator))) % 2**32)
    shape = tuple(reversed(in_size))
    if np.issubdtype(dtype, np.integer):
        arr = rng.integers(0, 3 if dtype == np.uint8 else 1000, size=shape).astype(dtype)
    else:
        arr = rng.normal(size=shape).astype(dtype)
    in_dir = rotation(in_dir) if in_dir else None
    out_dir = rotation(out_dir) if out_dir else tuple(np.eye(3).ravel())
    a, b = pair(sitk, shim, arr, in_sp, in_or, in_dir)
    interp = {
        "nearest": sitk.sitkNearestNeighbor,
        "linear": sitk.sitkLinear,
        "bspline": sitk.sitkBSpline,
    }[interpolator]
    ra, rb = _resample_both(
        sitk, shim, a, b, (out_size, out_sp, out_or, out_dir), interp, -7.0, nn, None
    )
    if interpolator == "bspline":
        assert_same(sitk, shim, ra, rb, atol=1e-5 * float(np.abs(arr).max()))
    elif interpolator == "linear" and np.issubdtype(dtype, np.integer):
        x, y = sitk.GetArrayFromImage(ra), shim.GetArrayFromImage(rb)
        assert np.mean(x != y) < 1e-3
        assert np.max(np.abs(x.astype(int) - y.astype(int))) <= 1
    else:
        assert_same(sitk, shim, ra, rb, rtol=1e-12, atol=1e-12)


def test_resample_nan_neighbors(sitk, shim):
    arr = np.random.default_rng(9).normal(size=(10, 11, 12))
    arr[3:5, :, 4] = np.nan
    a, b = pair(sitk, shim, arr, (3.0, 3.0, 3.0))
    geometry = ((12, 11, 10), (3.0, 3.0, 3.0), (0.0, 0.0, 0.0), tuple(np.eye(3).ravel()))
    ra, rb = _resample_both(sitk, shim, a, b, geometry, sitk.sitkLinear, 0.0, True, None)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(ra), shim.GetArrayFromImage(rb))
    geometry = ((20, 22, 18), (1.5, 1.5, 1.6), (0.5, 0.0, 0.0), tuple(np.eye(3).ravel()))
    ra, rb = _resample_both(sitk, shim, a, b, geometry, sitk.sitkLinear, 0.0, True, None)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(ra), shim.GetArrayFromImage(rb))


def test_resample_output_pixel_type_and_function_forms(sitk, shim):
    arr = np.random.default_rng(10).random((10, 11, 12)) * 300 - 20
    a, b = pair(sitk, shim, arr, (2.0, 2.0, 2.0), (1.0, 2.0, 3.0))
    ref = np.zeros((7, 9, 8), dtype=np.uint8)
    refa, refb = pair(sitk, shim, ref, (2.9, 2.3, 3.1), (0.0, 1.0, 2.0))
    geometry = ((8, 9, 7), (2.9, 2.3, 3.1), (0.0, 1.0, 2.0), tuple(np.eye(3).ravel()))
    ra, rb = _resample_both(
        sitk, shim, a, b, geometry, sitk.sitkLinear, 0.0, False, sitk.sitkUInt8
    )
    assert_same(sitk, shim, ra, rb)
    for args in [
        (refa, sitk.Transform(), sitk.sitkNearestNeighbor, 0),
        (refa, sitk.Transform(), sitk.sitkLinear, 5.0, sitk.sitkFloat32),
        (refa,),
    ]:
        args_b = tuple(
            refb if x is refa else shim.Transform() if isinstance(x, sitk.Transform) else x
            for x in args
        )
        assert_same(sitk, shim, sitk.Resample(a, *args), shim.Resample(b, *args_b), rtol=1e-6)
    fa, fb = sitk.ResampleImageFilter(), shim.ResampleImageFilter()
    fa.SetReferenceImage(refa)
    fb.SetReferenceImage(refb)
    assert_same(sitk, shim, fa.Execute(a), fb.Execute(b), rtol=1e-12)


def test_vector_resample(sitk, shim):
    arr = np.random.default_rng(11).random((6, 7, 8, 3))
    a, b = pair(sitk, shim, arr, (2.0, 2.0, 2.0), vector=True)
    geometry = ((9, 5, 7), (1.7, 2.9, 1.6), (0.3, 0.2, 0.1), tuple(np.eye(3).ravel()))
    ra, rb = _resample_both(sitk, shim, a, b, geometry, sitk.sitkLinear, 0.0, False, None)
    assert_same(sitk, shim, ra, rb, rtol=1e-12)


def test_2d_resample(sitk, shim):
    arr = np.random.default_rng(12).random((13, 17))
    a, b = pair(sitk, shim, arr, (1.1, 0.9), (2.0, -1.0))
    for interp in (sitk.sitkLinear, sitk.sitkNearestNeighbor):
        geometry = ((20, 11), (0.8, 1.3), (1.0, 0.0), (1.0, 0.0, 0.0, 1.0))
        ra, rb = _resample_both(sitk, shim, a, b, geometry, interp, 0.0, False, None)
        assert_same(sitk, shim, ra, rb, rtol=1e-12)


def test_unsupported_raises(shim):
    with pytest.raises(NotImplementedError, match="shim"):
        shim.ReadImage("x.mha")
    with pytest.raises(NotImplementedError):
        shim.Transform(3, 12)
    with pytest.raises(AttributeError, match="shim"):
        shim.GradientMagnitude  # noqa: B018
    assert not hasattr(shim, "GradientMagnitude")


def test_deepcopy(shim):
    import copy

    img = shim.GetImageFromArray(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    img.SetSpacing((1.0, 2.0, 3.0))
    img.SetMetaData("unit", "Gy")
    clone = copy.deepcopy(img)
    assert clone.GetSpacing() == img.GetSpacing()
    assert clone.GetMetaData("unit") == "Gy"
    np.testing.assert_array_equal(shim.GetArrayFromImage(clone), shim.GetArrayFromImage(img))
    assert clone._array is not img._array


@pytest.mark.parametrize("vector", [False, True])
def test_reduce_matches_simpleitk(sitk, shim, vector):
    """Same reduce tuple as SimpleITK; actual cross-runtime pickles: test_pickle_compat.py."""
    shape = (2, 3, 4, 2) if vector else (2, 3, 4)
    arr = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    a, b = pair(sitk, shim, arr, (1.0, 2.0, 3.0), (4.0, 5.0, 6.0), rotation(3), vector=vector)
    for img in (a, b):
        img.SetMetaData("unit", "Gy")
    cls_a, args_a, state_a = a.__reduce_ex__(4)
    cls_b, args_b, state_b = b.__reduce_ex__(4)
    assert (cls_b.__module__, cls_b.__qualname__) == (cls_a.__module__, cls_a.__qualname__)
    assert args_b == args_a
    assert state_b[0] == state_a[0] == 0
    assert state_b[1] == state_a[1]
    assert state_b[2:] == state_a[2:]
    restored = cls_b(*args_b)
    restored.__setstate__(state_a)
    assert_same(sitk, shim, a, restored)
    assert restored.GetMetaData("unit") == "Gy"


def test_tg119_geometry(sitk, shim):
    """Resample the TG119 masks and CT onto a 5 mm dose grid, as the pipeline does."""
    pytest.importorskip("pyRadPlan")
    from pyRadPlan import load_tg119

    ct, cst = load_tg119()
    hu = sitk.GetArrayFromImage(ct.cube_hu)
    spacing, origin = ct.cube_hu.GetSpacing(), ct.cube_hu.GetOrigin()
    a, b = pair(sitk, shim, hu, spacing, origin)
    grid = ct.grid.resample({"x": 5.0, "y": 5.0, "z": 5.0})
    geometry = (
        tuple(int(d) for d in grid.dimensions),
        tuple(float(v) for v in grid.resolution_vector),
        tuple(float(v) for v in grid.origin),
        tuple(float(v) for v in grid.direction.ravel()),
    )
    for interp, nn in itertools.product((sitk.sitkLinear, sitk.sitkNearestNeighbor), (0, 1)):
        ra, rb = _resample_both(sitk, shim, a, b, geometry, interp, 0.0, bool(nn), None)
        assert_same(sitk, shim, ra, rb, rtol=1e-12)
    for voi in cst.vois:
        mask = sitk.GetArrayFromImage(voi.mask)
        ma, mb = pair(sitk, shim, mask, spacing, origin)
        ra, rb = _resample_both(
            sitk, shim, ma, mb, geometry, sitk.sitkNearestNeighbor, 0.0, False, None
        )
        assert_same(sitk, shim, ra, rb)
