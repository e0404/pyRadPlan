"""The Array API collapsed cone transport against an independent NumPy reference."""

import array_api_compat
import numpy as np
import pytest

from pyRadPlan.core.xp_utils import cupy_available, to_numpy
from pyRadPlan.dose.engines._collapsed_cone import transport


def _skip_unavailable(name):
    """Skip the test if the backend (or its GPU) is not available."""
    module = pytest.importorskip("torch" if name.startswith("torch") else name)
    if name == "cupy" and not cupy_available():
        pytest.skip("no CuPy with a CUDA device")
    if name == "torch_cuda" and not module.cuda.is_available():
        pytest.skip("no CUDA device for torch")
    return module


@pytest.fixture(params=["numpy", "array_api_strict", "torch", "cupy", "torch_cuda"])
def backend_name(request):
    _skip_unavailable(request.param)
    return request.param


@pytest.fixture
def xp(backend_name):
    module = _skip_unavailable(backend_name)
    return array_api_compat.array_namespace(module.asarray([0.0]))


@pytest.fixture
def device(backend_name):
    return "cuda" if backend_name == "torch_cuda" else None


# Independent reference: direction by direction, vectorized over the lattice lines of each plane
# with index gathers (the formulation of matRad's matRad_collapsedConeTransport)
def _reference_transport(terma, rho, voxel_size, dirs, weights, att, origin=(0, 0, 0)):
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


def _directions(n):
    """Fixed, slightly perturbed unit directions (as in matRad's transport tests)."""
    base = np.array(
        [
            [0.3, -0.7, 0.5],
            [-0.8, 0.1, 0.6],
            [0.2, 0.9, -0.4],
            [1, 0, 0],
            [0, 0, -1],
            [0.577, 0.577, 0.577],
        ]
    ).T
    dirs = np.tile(base, (1, int(np.ceil(n / base.shape[1]))))[:, :n]
    dirs = dirs + 0.05 * np.sin(np.arange(1, 3 * n + 1)).reshape(n, 3).T
    return dirs / np.linalg.norm(dirs, axis=0)


def _kernel(n_dirs):
    """Two kernel components (primary and scatter like) on n_dirs directions."""
    weights = np.vstack([np.linspace(0.01, 0.05, n_dirs), np.linspace(0.03, 0.01, n_dirs)])
    att = np.vstack([np.linspace(0.2, 1.5, n_dirs), np.linspace(0.002, 0.02, n_dirs)])
    return weights, att


@pytest.fixture
def heterogeneous_input():
    """Heterogeneous TERMA and density with two kernel components on 16 directions."""
    i, j, k = np.meshgrid(np.arange(1, 14), np.arange(1, 18), np.arange(1, 12), indexing="ij")
    terma = np.exp(-0.1 * i) * (1 + 0.5 * np.sin(j)) * (1 + 0.2 * np.cos(k))
    rho = 0.6 + 0.5 * np.sin(0.7 * i + 0.3 * j) * np.cos(0.5 * k)
    rho[4:7, 5:9, 3:6] = 0
    return (terma, rho, np.array([2.0, 3.0, 2.5]), _directions(16), *_kernel(16))


def _case(name):
    """Transport inputs and origin of the edge cases compared with the reference."""
    rng = np.random.default_rng(1)
    shape = (12, 15, 10)
    terma = rng.random(shape)
    rho = 0.5 + rng.random(shape)
    voxel_size = np.array([2.0, 3.0, 2.5])
    dirs = _directions(16)
    origin = (0, 0, 0)
    if name == "anisotropic":
        voxel_size = np.array([2.0, 3.0, 5.0])
        origin = (4, -7, 2)
    elif name == "low_density":
        rho = 1e-6 * (1 + rng.random(shape))
    elif name == "zero_density":
        rho[3:8, 4:11, 2:7] = 0
        rho[:, 0, :] = 0
    elif name == "off_centre_source":
        terma = np.zeros(shape)
        terma[9, 2, 7] = 1.0
        terma[10, 2, 7] = 0.5
        origin = (11, 3, -5)
    elif name == "tie_break":
        # |d| / s is equal along the first two axes: the first one is dominant
        voxel_size = np.array([1.0, 2.0, 1.0])
        tie = np.array([1.0, 2.0, 0.3]) / np.linalg.norm([1.0, 2.0, 0.3])
        assert np.argmax(np.abs(tie) / voxel_size) == 0
        dirs = np.column_stack([tie, -tie, [-1.0, 2.0, 0.3] / np.linalg.norm([1.0, 2.0, 0.3])])
    return terma, rho, voxel_size, dirs, *_kernel(dirs.shape[1]), origin


_CASES = ["heterogeneous", "anisotropic", "low_density", "zero_density", "off_centre_source"]


@pytest.mark.parametrize("case", [*_CASES, "tie_break"])
def test_transport_matches_reference(xp, device, case):
    """The Array API transport equals the independent NumPy reference on every backend."""
    terma, rho, voxel_size, dirs, weights, att, origin = _case(case)
    ref = _reference_transport(terma, rho, voxel_size, dirs, weights, att, origin)
    dose = transport(
        xp.asarray(terma, device=device),
        xp.asarray(rho, device=device),
        voxel_size,
        dirs,
        weights,
        att,
        origin,
    )
    assert array_api_compat.array_namespace(dose) is xp
    # The rounding errors of both recursions scale with the released energy, which at very
    # low density is far above the deposited dose
    scale = max(ref.max(), terma.max() * weights.sum())
    np.testing.assert_allclose(to_numpy(dose), ref, rtol=0, atol=1e-12 * scale)


@pytest.mark.parametrize("origin", [(0, 0, 0), (5, -3, 7)])
def test_transport_heterogeneous_matches_reference(xp, device, heterogeneous_input, origin):
    """Heterogeneous TERMA and density with void voxels, also on a shifted lattice."""
    terma, rho, *kernel = heterogeneous_input
    ref = _reference_transport(terma, rho, *kernel, origin=origin)
    dose = transport(
        xp.asarray(terma, device=device), xp.asarray(rho, device=device), *kernel, origin=origin
    )
    np.testing.assert_allclose(to_numpy(dose), ref, rtol=0, atol=1e-12 * ref.max())


def test_transport_single_precision(xp, device, heterogeneous_input):
    """A float32 TERMA gives a float32 dose close to the double precision result."""
    terma, rho, *kernel = heterogeneous_input
    ref = _reference_transport(terma, rho, *kernel)
    dose = transport(
        xp.astype(xp.asarray(terma, device=device), xp.float32),
        xp.astype(xp.asarray(rho, device=device), xp.float32),
        *kernel,
    )
    assert dose.dtype == xp.float32
    np.testing.assert_allclose(to_numpy(dose), ref, rtol=0, atol=1e-5 * ref.max())


def test_transport_finite_and_non_negative(heterogeneous_input):
    """The transport gives a finite, non-negative dose of the TERMA's shape."""
    dose = transport(*heterogeneous_input)
    assert dose.shape == heterogeneous_input[0].shape
    assert np.all(np.isfinite(dose))
    assert np.all(dose >= 0)
    assert np.any(dose > 0)


def test_transport_without_terma_is_zero(xp, device):
    """No TERMA gives no dose."""
    zeros = xp.asarray(np.zeros((4, 5, 6)), device=device)
    dose = transport(zeros, zeros + 1, [1, 1, 1], _directions(3), *_kernel(3))
    assert not bool(xp.any(dose != 0))


def test_transport_sub_volume_consistency(xp, device, heterogeneous_input):
    """A sub-volume with its origin gives the dose of the full volume inside it."""
    terma, rho, voxel_size, dirs, weights, att = heterogeneous_input
    terma = np.zeros_like(terma)
    terma[5:8, 6:10, 4:7] = 1 + np.arange(1, 37).reshape(3, 4, 3, order="F") / 36
    kernel = (voxel_size, dirs, weights, att)
    full = to_numpy(
        transport(xp.asarray(terma, device=device), xp.asarray(rho, device=device), *kernel)
    )
    box = (slice(2, 12), slice(3, 15), slice(1, 10))
    sub = transport(
        xp.asarray(terma[box], device=device),
        xp.asarray(rho[box], device=device),
        *kernel,
        origin=(2, 3, 1),
    )
    np.testing.assert_allclose(to_numpy(sub), full[box], rtol=0, atol=1e-15)


def test_transport_component_linearity(heterogeneous_input):
    """The dose of several kernel components is the sum of their single doses."""
    terma, rho, voxel_size, dirs, weights, att = heterogeneous_input
    w = np.vstack([weights, 0.5 * weights[1]])
    a = np.vstack([att, 3 * att[1]])
    dose_all = transport(terma, rho, voxel_size, dirs, w, a)
    dose_sum = sum(transport(terma, rho, voxel_size, dirs, w[[c]], a[[c]]) for c in range(3))
    np.testing.assert_allclose(dose_all, dose_sum, rtol=0, atol=1e-12 * dose_all.max())


def test_transport_energy_conservation_short_range():
    """A short-range kernel deposits the released energy inside the volume."""
    rng = np.random.default_rng(0)
    terma = np.zeros((41, 41, 41))
    terma[15:26, 15:26, 15:26] = 1 + rng.random((11, 11, 11))
    weights = np.vstack([0.05 * np.ones(12), 0.03 * np.ones(12)])
    att = np.vstack([2 * np.ones(12), np.ones(12)])
    dose = transport(terma, np.ones_like(terma), [2, 2, 2], _directions(12), weights, att)
    np.testing.assert_allclose(dose.sum(), terma.sum() * weights.sum(), rtol=1e-6)


def test_transport_equilibrium_each_voxel_visited_once():
    """Uniform TERMA along one direction reaches equilibrium at the kernel weight."""
    terma = np.ones((30, 30, 30))
    for d in _directions(6).T:
        dose = transport(
            terma, np.ones_like(terma), [2, 2, 2], d.reshape(3, 1), [[0.1], [0.0]], [[3], [1]]
        )
        assert dose.max() <= 0.1 * (1 + 1e-12)
        np.testing.assert_allclose(np.median(dose), 0.1, rtol=1e-6)


def test_transport_density_scaling():
    """Half the density on a twice as large voxel gives the same dose."""
    terma = np.zeros((60, 5, 5))
    terma[4, 2, 2] = 1
    dirs = np.array([[1.0], [0.0], [0.0]])
    water = transport(terma, np.ones_like(terma), [1, 1, 1], dirs, [[0], [1]], [[1], [0.1]])
    half = transport(terma, 0.5 * np.ones_like(terma), [2, 1, 1], dirs, [[0], [1]], [[1], [0.1]])
    np.testing.assert_allclose(half[:, 2, 2], water[:, 2, 2], rtol=1e-12)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_transport_axis_mirror_symmetry(axis):
    """Opposite directions along an axis give mirrored doses."""
    terma = np.zeros((25, 21, 19))
    terma[12, 10, 9] = 1
    direction = np.zeros((3, 1))
    direction[axis] = 1
    args = ([2, 3, 2.5],)
    kernel = ([[0.1], [0.05]], [[0.3], [0.05]])
    fwd = transport(terma, np.ones_like(terma), *args, direction, *kernel)
    bwd = transport(terma, np.ones_like(terma), *args, -direction, *kernel)
    np.testing.assert_allclose(bwd, np.flip(fwd, axis), rtol=0, atol=1e-15)
