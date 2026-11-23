import ctypes
import shutil
import subprocess

import numpy as np
import pytest
from pymatreader import read_mat
import scipy.io
import SimpleITK as sitk

from pyRadPlan import PhotonPlan, generate_stf
from pyRadPlan.ct import create_ct
from pyRadPlan.cst import create_cst
from pyRadPlan.dose import calc_dose_forward, calc_dose_influence
from pyRadPlan.dose.engines import PhotonCollapsedConeEngine, get_engine
from pyRadPlan.dose.engines import _collapsed_cone as cc
from pyRadPlan.dose.engines import _collapsed_cone_cuda as cuda
from pyRadPlan.machines import load_from_name


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


@pytest.fixture
def heterogeneous_input():
    """Heterogeneous TERMA and density with two kernel components on 16 directions."""
    i, j, k = np.meshgrid(np.arange(1, 14), np.arange(1, 18), np.arange(1, 12), indexing="ij")
    terma = np.exp(-0.1 * i) * (1 + 0.5 * np.sin(j)) * (1 + 0.2 * np.cos(k))
    rho = 0.6 + 0.5 * np.sin(0.7 * i + 0.3 * j) * np.cos(0.5 * k)
    rho[4:7, 5:9, 3:6] = 0
    weights = np.vstack([np.linspace(0.01, 0.05, 16), np.linspace(0.03, 0.01, 16)])
    att = np.vstack([np.linspace(0.2, 1.5, 16), np.linspace(0.002, 0.02, 16)])
    return terma, rho, np.array([2.0, 3.0, 2.5]), _directions(16), weights, att


def test_transport_finite_and_non_negative(heterogeneous_input):
    """The NumPy transport gives a finite, non-negative dose of the TERMA's shape."""
    dose = cc.transport(*heterogeneous_input, use_numba=False)
    assert dose.shape == heterogeneous_input[0].shape
    assert np.all(np.isfinite(dose))
    assert np.all(dose >= 0)
    assert np.any(dose > 0)


@pytest.mark.skipif(not cc.NUMBA_AVAILABLE, reason="numba not installed")
@pytest.mark.parametrize("origin", [(0, 0, 0), (5, -3, 7)])
def test_transport_numba_matches_numpy(heterogeneous_input, origin):
    """The numba transport equals the NumPy reference, also on a shifted lattice."""
    ref = cc.transport(*heterogeneous_input, origin=origin, use_numba=False)
    jit = cc.transport(*heterogeneous_input, origin=origin, use_numba=True, num_threads=2)
    np.testing.assert_allclose(jit, ref, rtol=0, atol=1e-12 * ref.max())


@pytest.mark.parametrize("use_numba", [False, True])
def test_transport_sub_volume_consistency(heterogeneous_input, use_numba):
    """A sub-volume with its origin gives the dose of the full volume inside it."""
    terma, rho, voxel_size, dirs, weights, att = heterogeneous_input
    terma = np.zeros_like(terma)
    terma[5:8, 6:10, 4:7] = 1 + np.arange(1, 37).reshape(3, 4, 3, order="F") / 36
    full = cc.transport(terma, rho, voxel_size, dirs, weights, att, use_numba=use_numba)
    box = (slice(2, 12), slice(3, 15), slice(1, 10))
    sub = cc.transport(
        terma[box],
        rho[box],
        voxel_size,
        dirs,
        weights,
        att,
        origin=(2, 3, 1),
        use_numba=use_numba,
    )
    np.testing.assert_allclose(sub, full[box], rtol=0, atol=1e-15)


def test_transport_component_linearity(heterogeneous_input):
    """The dose of several kernel components is the sum of their single doses."""
    terma, rho, voxel_size, dirs, weights, att = heterogeneous_input
    w = np.vstack([weights, 0.5 * weights[1]])
    a = np.vstack([att, 3 * att[1]])
    dose_all = cc.transport(terma, rho, voxel_size, dirs, w, a, use_numba=False)
    dose_sum = sum(
        cc.transport(terma, rho, voxel_size, dirs, w[[c]], a[[c]], use_numba=False)
        for c in range(3)
    )
    np.testing.assert_allclose(dose_all, dose_sum, rtol=0, atol=1e-12 * dose_all.max())


def test_transport_energy_conservation_short_range():
    """A short-range kernel deposits the released energy inside the volume."""
    rng = np.random.default_rng(0)
    terma = np.zeros((41, 41, 41))
    terma[15:26, 15:26, 15:26] = 1 + rng.random((11, 11, 11))
    weights = np.vstack([0.05 * np.ones(12), 0.03 * np.ones(12)])
    att = np.vstack([2 * np.ones(12), np.ones(12)])
    dose = cc.transport(terma, np.ones_like(terma), [2, 2, 2], _directions(12), weights, att)
    np.testing.assert_allclose(dose.sum(), terma.sum() * weights.sum(), rtol=1e-6)


def test_transport_equilibrium_each_voxel_visited_once():
    """Uniform TERMA along one direction reaches equilibrium at the kernel weight."""
    terma = np.ones((30, 30, 30))
    for d in _directions(6).T:
        dose = cc.transport(
            terma, np.ones_like(terma), [2, 2, 2], d.reshape(3, 1), [[0.1], [0.0]], [[3], [1]]
        )
        assert dose.max() <= 0.1 * (1 + 1e-12)
        np.testing.assert_allclose(np.median(dose), 0.1, rtol=1e-6)


def test_transport_density_scaling():
    """Half the density on a twice as large voxel gives the same dose."""
    terma = np.zeros((60, 5, 5))
    terma[4, 2, 2] = 1
    dirs = np.array([[1.0], [0.0], [0.0]])
    water = cc.transport(terma, np.ones_like(terma), [1, 1, 1], dirs, [[0], [1]], [[1], [0.1]])
    half = cc.transport(
        terma, 0.5 * np.ones_like(terma), [2, 1, 1], dirs, [[0], [1]], [[1], [0.1]]
    )
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
    fwd = cc.transport(terma, np.ones_like(terma), *args, direction, *kernel)
    bwd = cc.transport(terma, np.ones_like(terma), *args, -direction, *kernel)
    np.testing.assert_allclose(bwd, np.flip(fwd, axis), rtol=0, atol=1e-15)


def _synthetic_edk():
    """Angle and radius edges of a synthetic conical kernel and its inner and outer radii."""
    radius_edges = 10 * np.array(
        [
            0,
            0.05,
            0.1,
            0.15,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.8,
            1,
            1.5,
            2,
            3,
            4,
            5,
            6,
            8,
            10,
            15,
            20,
            30,
            40,
            50,
            60,
        ]
    )
    return np.arange(0, 180.01, 3.75), radius_edges, radius_edges[:-1], radius_edges[1:]


def test_fit_kernel_synthetic():
    """The fit recovers the attenuation coefficients and energies of a synthetic kernel."""
    angle_edges, radius_edges, r1, r2 = _synthetic_edk()
    theta = 0.5 * (angle_edges[:-1] + angle_edges[1:])
    cone_a = 0.02 * np.cos(np.deg2rad(theta / 2)) ** 4
    cone_b = 0.01 * np.ones_like(theta)
    edk = {
        "angleEdges": angle_edges,
        "radiusEdges": radius_edges,
        "primary": np.outer(cone_a, np.exp(-0.3 * r1) - np.exp(-0.3 * r2)),
        "scatter": np.outer(cone_b, np.exp(-0.02 * r1) - np.exp(-0.02 * r2)),
    }
    kernel = cc.fit_collapsed_cone_kernel(edk, 12)
    assert kernel.num_polar_angles == 12
    np.testing.assert_allclose(kernel.polar_edges, np.arange(0, 181, 15))
    np.testing.assert_allclose(kernel.A.sum(), edk["primary"].sum(), rtol=1e-12)
    np.testing.assert_allclose(kernel.B.sum(), edk["scatter"].sum(), rtol=1e-12)
    np.testing.assert_allclose(kernel.a, 0.3, rtol=1e-4)
    np.testing.assert_allclose(kernel.b, 0.02, rtol=1e-3)
    assert np.all(kernel.residual["primaryCum"] < 1e-4)

    kernel = cc.fit_collapsed_cone_kernel(edk, 12, 1.5, 1, 200)
    assert kernel.min_radius == 1.5
    assert kernel.scatter_max_radius == 200
    np.testing.assert_allclose(kernel.b, 0.02, rtol=1e-3)


def test_fit_kernel_two_scatter_terms():
    """The two-term fit recovers the faster scatter coefficient."""
    angle_edges, radius_edges, r1, r2 = _synthetic_edk()

    def shell(a):
        return np.exp(-a * r1) - np.exp(-a * r2)

    edk = {
        "angleEdges": angle_edges,
        "radiusEdges": radius_edges,
        "primary": 0.02 * np.outer(np.ones(48), shell(0.3)),
        "scatter": np.outer(np.ones(48), 0.006 * shell(0.1) + 0.004 * shell(0.02)),
    }
    kernel = cc.fit_collapsed_cone_kernel(edk, 12, 0, 2)
    assert kernel.B.shape == (2, 12)
    np.testing.assert_allclose(kernel.B.sum(), edk["scatter"].sum(), rtol=1e-10)
    np.testing.assert_allclose(kernel.b[0], 0.1, rtol=0.02)


def test_default_kernel_data():
    """The 6 MV kernel is valid and its energy does not change with refitting."""
    kernel, meta = cc.load_kernel(None, 12)
    assert kernel.A.shape == (1, 12)
    assert np.all((kernel.A >= 0) & (kernel.B >= 0) & (kernel.a > 0) & (kernel.b > 0))
    total = kernel.A.sum() + kernel.B.sum()
    assert 0.9 < total < 1
    assert meta["nominalEnergy"] == 6
    kernel6, _ = cc.load_kernel(None, 6)
    np.testing.assert_allclose(kernel6.A.sum() + kernel6.B.sum(), total, rtol=1e-10)
    kernel2, _ = cc.load_kernel(None, 12, 2)
    assert kernel2.B.shape == (2, 12)
    np.testing.assert_allclose(kernel2.A.sum() + kernel2.B.sum(), total, rtol=1e-10)


def test_hardening_fit():
    """The hardening correction matches the machine depth dose better than a constant mu."""
    machine = load_from_name("photons", "Generic")
    svd = machine.get_kernel_by_energy(6.0)
    ix = int(np.argmin(np.abs(svd.kernel_ssds - machine.sad)))

    def reference(d):
        return cc.analytical_depth_dose(
            svd.kernel_data[ix], svd.kernel_pos, svd.kernel_betas, svd.m, d
        )

    kernel, _ = cc.load_kernel(None, 12)
    mu0, beta = cc.fit_hardening(kernel, reference, svd.m, 5, 100)
    assert mu0 > 0 and beta > 0
    depths = np.array([30.0, 100.0, 200.0, 300.0])
    ref = reference(depths) / reference(depths)[1]
    corrected = cc.broad_field_depth_dose(kernel, mu0, depths, 5, beta)
    corrected = corrected / corrected[1]
    mono = cc.broad_field_depth_dose(kernel, svd.m, depths, 5, 0)
    mono = mono / mono[1]
    assert np.max(np.abs(corrected / ref - 1)) < 0.02
    assert np.max(np.abs(corrected / ref - 1)) < np.max(np.abs(mono / ref - 1))
    np.testing.assert_allclose(
        cc.terma_depth_factor(0.005, 0, depths), 0.005 * np.exp(-0.005 * depths), rtol=1e-14
    )


def test_check_kernel_energy():
    """Matching, mismatching and unknown kernel energies."""
    assert cc.check_kernel_energy({"nominalEnergy": 6.0}, 6.0) == (True, "")
    ok, msg = cc.check_kernel_energy({"nominalEnergy": 6.0}, 18.0)
    assert not ok and "18" in msg
    ok, msg = cc.check_kernel_energy({}, 6.0)
    assert ok and msg


def test_engine_from_pln():
    """The plan selects the engine and sets its properties."""
    pln = PhotonPlan(machine="Generic")
    pln.prop_dose_calc = {
        "engine": "CCC",
        "num_azimuthal_angles": 6,
        "scatter_cutoff": 80.0,
    }
    engine = get_engine(pln)
    assert isinstance(engine, PhotonCollapsedConeEngine)
    assert engine.num_azimuthal_angles == 6
    assert engine.scatter_cutoff == 80
    assert engine.num_polar_angles == 12
    assert engine.kernel_tilt == "beam"


def test_is_available():
    """The engine is available with the default kernel, not with a missing kernel file."""
    machine = load_from_name("photons", "Generic")
    pln = PhotonPlan(machine="Generic")
    available, _ = PhotonCollapsedConeEngine.is_available(pln, machine)
    assert available
    pln.prop_dose_calc = {"kernel_file": "does_not_exist.mat"}
    available, msg = PhotonCollapsedConeEngine.is_available(pln, machine)
    assert not available and "not found" in msg


def _water_box(num_vox, res, target_size, low_density_rows=None):
    """Water cube in air of num_vox^3 voxels with a centered cubic target."""
    hu = np.full((num_vox, num_vox, num_vox), -1000, dtype=np.int16)
    hu[2:-2, 2:-2, 2:-2] = 0
    if low_density_rows is not None:
        hu[2:-2, low_density_rows, 2:-2] = -700
    img = sitk.GetImageFromArray(hu)
    img.SetSpacing((res, res, res))
    ct = create_ct(cube_hu=img)
    c = num_vox // 2
    half = target_size // 2
    target = np.zeros((num_vox,) * 3, dtype=bool)
    target[c - half : c + half, c - half : c + half, c - half : c + half] = True
    body = np.zeros_like(target)
    body[2:-2, 2:-2, 2:-2] = True
    cst = create_cst(
        [
            [
                0,
                "Water",
                "OAR",
                np.flatnonzero(body.transpose(1, 2, 0).ravel(order="F")) + 1.0,
                2,
                [],
            ],
            [
                1,
                "Target",
                "TARGET",
                np.flatnonzero(target.transpose(1, 2, 0).ravel(order="F")) + 1.0,
                1,
                [],
            ],
        ],
        ct=ct,
    )
    return ct, cst


def _plan(res, bixel_width, gantry_angles, **dose_calc):
    """Single-energy CCC photon plan with a reduced number of cones."""
    pln = PhotonPlan(machine="Generic")
    pln.prop_stf = {
        "gantry_angles": list(gantry_angles),
        "couch_angles": [0.0] * len(gantry_angles),
        "bixel_width": bixel_width,
    }
    pln.prop_dose_calc = {
        "engine": "CCC",
        "dose_grid": {"resolution": {"x": res, "y": res, "z": res}},
        "num_polar_angles": 4,
        "num_azimuthal_angles": 4,
        "scatter_cutoff": 50.0,
        "enable_dij_sampling": False,
        **dose_calc,
    }
    return pln


def _dij_times_weights(dij, w, shape):
    """Dose of the weights from the first dose influence matrix."""
    return np.asarray(dij.physical_dose.flat[0] @ w).reshape(shape)


def test_forward_equals_dij_times_weights():
    """The bixel-wise forward dose equals the dose influence matrix times the weights."""
    ct, cst = _water_box(16, 6.0, 4)
    pln = _plan(6.0, 12.0, [0.0], beam_forward=False)
    stf = generate_stf(ct, cst, pln)
    dij = calc_dose_influence(ct, cst, stf, pln)
    w = np.linspace(0.5, 1.5, dij.total_num_of_bixels)
    dose = sitk.GetArrayFromImage(calc_dose_forward(ct, cst, stf, pln, w).physical_dose.values)
    ref = _dij_times_weights(dij, w, dose.shape)
    np.testing.assert_allclose(dose, ref, rtol=0, atol=1e-5 * ref.max())
    assert np.all(dose >= 0) and np.all(np.isfinite(dose))
    assert dose.max() > 0


def test_beam_forward_equals_dij_times_weights():
    """Without cut-offs the beam forward dose equals the dij times the weights."""
    ct, cst = _water_box(16, 6.0, 4, low_density_rows=slice(6, 9))
    pln = _plan(6.0, 12.0, [0.0, 120.0], scatter_cutoff=1000.0, geometric_lateral_cutoff=1000.0)
    stf = generate_stf(ct, cst, pln)
    dij = calc_dose_influence(ct, cst, stf, pln)
    w = np.linspace(0.5, 1.5, dij.total_num_of_bixels)
    dose = sitk.GetArrayFromImage(calc_dose_forward(ct, cst, stf, pln, w).physical_dose.values)
    ref = _dij_times_weights(dij, w, dose.shape)
    np.testing.assert_allclose(dose, ref, rtol=0, atol=1e-5 * ref.max())


def test_short_range_kernel_central_axis(tmp_path):
    """With a kernel depositing locally the dose follows the TERMA along the central axis."""
    n_polar = 4
    edges = np.linspace(0, 180, n_polar + 1)
    cos_edges = np.cos(np.deg2rad(edges))
    solid_angle = (cos_edges[:-1] - cos_edges[1:]) / 2
    kernel_file = tmp_path / "short_range_kernel.mat"
    scipy.io.savemat(
        kernel_file,
        {
            "ccKernel": {
                "fit": {
                    "numPolarAngles": float(n_polar),
                    "polarEdges": edges,
                    "polarAngles": np.rad2deg(np.arccos(0.5 * (cos_edges[:-1] + cos_edges[1:]))),
                    "A": 0.6 * solid_angle,
                    "a": 5 * np.ones(n_polar),
                    "B": 0.3 * solid_angle,
                    "b": 3 * np.ones(n_polar),
                }
            }
        },
    )
    ct, cst = _water_box(30, 6.0, 10)
    pln = _plan(6.0, 20.0, [0.0], kernel_file=str(kernel_file), scatter_cutoff=20.0)
    stf = generate_stf(ct, cst, pln)
    engine = get_engine(pln)
    image = engine.calc_dose_forward(
        ct, cst, stf, np.ones(stf.total_number_of_bixels)
    ).physical_dose.values
    dose = sitk.GetArrayFromImage(image)
    assert np.all(dose >= 0) and np.all(np.isfinite(dose))

    params = next(iter(engine._energy_params.values()))
    machine = load_from_name("photons", "Generic")
    source = np.asarray(stf.beams[0].iso_center) + np.asarray(stf.beams[0].source_point)
    # Gantry 0: the beam runs along the world y axis, i.e. numpy axis 1
    c = dose.shape[0] // 2
    column = dose[c, :, c]
    if source[1] > 0:
        column = column[::-1]
    first = int(np.flatnonzero(column > 0)[0])
    rows = np.arange(first + 3, first + 21)
    depth = (rows - first + 0.5) * 6.0
    idx = rows if source[1] < 0 else dose.shape[1] - 1 - rows
    world = np.array([image.TransformIndexToPhysicalPoint((int(c), int(i), int(c))) for i in idx])
    inv_square = (machine.sad / np.linalg.norm(world - source, axis=1)) ** 2
    terma = cc.terma_depth_factor(params["mu0"], params["beta"], depth) * inv_square
    expected = params["normalization"] * (0.6 + 0.3) * terma
    np.testing.assert_allclose(column[rows], expected, rtol=0.01)


@pytest.mark.parametrize("gantry_angle", [0.0, 90.0])
def test_depth_dose_matches_svd(gantry_angle):
    """In water the central axis depth dose of a broad beam agrees with the SVD engine."""
    ct, cst = _water_box(40, 5.0, 16)
    doses = {}
    for engine in ("SVDPB", "CCC"):
        pln = _plan(5.0, 10.0, [gantry_angle], engine=engine, scatter_cutoff=100.0)
        stf = generate_stf(ct, cst, pln)
        doses[engine] = sitk.GetArrayFromImage(
            calc_dose_forward(ct, cst, stf, pln).physical_dose.values
        )
    c = 20
    axis = (slice(None), c) if gantry_angle == 0.0 else (c, slice(None))
    svd = doses["SVDPB"][c][axis]
    ccc = doses["CCC"][c][axis]
    mask = svd > 0.5 * svd.max()
    assert mask.sum() > 10
    np.testing.assert_allclose(ccc[mask] / svd[mask], 1.0, atol=0.08)


def test_kernel_energy_mismatch_raises(tmp_path):
    """A kernel of another energy than the beam is refused."""
    data = read_mat(str(cc.DEFAULT_KERNEL_FILE))
    data["ccKernel"]["meta"]["nominalEnergy"] = 18.0
    kernel_file = tmp_path / "kernel_18MV.mat"
    scipy.io.savemat(kernel_file, {"ccKernel": data["ccKernel"]})
    ct, cst = _water_box(16, 6.0, 4)
    pln = _plan(6.0, 12.0, [0.0], kernel_file=str(kernel_file))
    stf = generate_stf(ct, cst, pln)
    with pytest.raises(ValueError, match="does not match"):
        calc_dose_forward(ct, cst, stf, pln)
    machine = load_from_name("photons", "Generic")
    available, msg = PhotonCollapsedConeEngine.is_available(pln, machine)
    assert not available and "does not match" in msg


_HOST_DRIVER = r"""
extern "C" void host_direction(ll num_lines, const double* terma, const double* rho,
    double* dose, const ll* sh1, const ll* sh2, const ll* bounds, ll nk, ll n1, ll n2,
    ll sk, ll s1, ll s2, ll ul1, ll uh1, ll ul2, ll uh2, ll ml1, ll ml2, ll nl1,
    double step, int forward, int nc, const double* w, const double* att) {
    for (ll line = 0; line < num_lines; ++line)
        ccc_line(line, terma, rho, dose, sh1, sh2, bounds, nk, n1, n2, sk, s1, s2, ul1, uh1,
                 ul2, uh2, ml1, ml2, nl1, step, forward, nc, w, att);
}
"""


@pytest.mark.skipif(shutil.which("g++") is None, reason="no C++ compiler")
def test_cuda_kernel_source_matches_cpu_transport(tmp_path, heterogeneous_input):
    """The CUDA kernel source, compiled for the host one line after the other."""
    source = tmp_path / "ccc_host.cpp"
    source.write_text("#define CCC_HOST\n" + cuda.KERNEL_SOURCE + _HOST_DRIVER)
    library = tmp_path / "ccc_host.so"
    subprocess.run(["g++", "-O2", "-shared", "-fPIC", "-o", str(library), str(source)], check=True)
    lib = ctypes.CDLL(str(library))
    ptr = np.ctypeslib.ndpointer
    i64, f64, i32 = ctypes.c_longlong, ctypes.c_double, ctypes.c_int
    lib.host_direction.argtypes = (
        [i64, ptr(np.float64), ptr(np.float64), ptr(np.float64)]
        + [ptr(np.int64)] * 3
        + [i64] * 13
        + [f64, i32, i32, ptr(np.float64), ptr(np.float64)]
    )

    terma, rho, voxel_size, dirs, weights, att = heterogeneous_input
    terma = np.ascontiguousarray(terma)
    terma[:, :, :3] = 0
    origin = np.array([5, -3, 7])
    dose = np.zeros_like(terma)
    bounds, union = cuda.source_bounds(np.stack(np.nonzero(terma)), terma.shape)
    for d in range(dirs.shape[1]):
        s = cuda.direction_setup(terma.shape, voxel_size, origin, dirs[:, d], union)
        k = s["k"]
        lib.host_direction(
            s["num_lines1"] * s["num_lines2"], terma, np.ascontiguousarray(rho), dose,
            s["sh1"], s["sh2"], np.ascontiguousarray(bounds[k].ravel()), s["nk"], s["n1"],
            s["n2"], *s["strides"], *union[k], s["min_line1"], s["min_line2"],
            s["num_lines1"], s["step"], s["forward"], weights.shape[0],
            np.ascontiguousarray(weights[:, d]), np.ascontiguousarray(att[:, d]),
        )  # fmt: skip
    ref = cc.transport(terma, rho, voxel_size, dirs, weights, att, origin=origin)
    np.testing.assert_allclose(dose, ref, rtol=0, atol=1e-12 * ref.max())


@pytest.mark.skipif(not cuda.gpu_available(), reason="no CuPy with a CUDA device")
def test_gpu_transport_matches_cpu(heterogeneous_input):
    """The CUDA transport equals the CPU transport."""
    cupy = pytest.importorskip("cupy")
    terma, rho, voxel_size, dirs, weights, att = heterogeneous_input
    ref = cc.transport(terma, rho, voxel_size, dirs, weights, att, origin=(5, -3, 7))
    gpu = cc.transport(
        cupy.asarray(terma), cupy.asarray(rho), voxel_size, dirs, weights, att, origin=(5, -3, 7)
    )
    np.testing.assert_allclose(cupy.asnumpy(gpu), ref, rtol=0, atol=1e-12 * ref.max())


@pytest.mark.skipif(not cuda.gpu_available(), reason="no CuPy with a CUDA device")
@pytest.mark.parametrize("beam_forward", [True, False])
def test_gpu_engine_matches_cpu(beam_forward):
    """The engine gives the same dose on the GPU and on the CPU."""
    ct, cst = _water_box(16, 6.0, 4, low_density_rows=slice(6, 9))
    doses = []
    for use_gpu in (False, True):
        pln = _plan(6.0, 12.0, [0.0, 120.0], use_gpu=use_gpu, beam_forward=beam_forward)
        stf = generate_stf(ct, cst, pln)
        doses.append(
            sitk.GetArrayFromImage(calc_dose_forward(ct, cst, stf, pln).physical_dose.values)
        )
    np.testing.assert_allclose(doses[1], doses[0], rtol=0, atol=1e-5 * doses[0].max())
