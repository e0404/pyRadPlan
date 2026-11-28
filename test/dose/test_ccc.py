"""Photon collapsed cone engine: TERMA, hardening fit and dose calculation."""

import array_api_compat
import numpy as np
import pytest
from pymatreader import read_mat
import scipy.io
import SimpleITK as sitk

from pyRadPlan import PhotonPlan, generate_stf, settings
from pyRadPlan.core.xp_utils import cupy_available, to_numpy
from pyRadPlan.ct import create_ct
from pyRadPlan.cst import create_cst
from pyRadPlan.dose import calc_dose_forward, calc_dose_influence
from pyRadPlan.dose.engines import PhotonCollapsedConeEngine, get_engine
from pyRadPlan.dose.engines._collapsed_cone import (
    analytical_depth_dose,
    broad_field_depth_dose,
    calc_terma,
    fit_hardening,
    terma_depth_factor,
)
from pyRadPlan.machines import load_from_name
from pyRadPlan.machines.photons._collapsed_cone_fit import load_kernel
from pyRadPlan.machines.photons._collapsed_cone_kernel import DEFAULT_KERNEL_FILE


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


def test_calc_terma_matches_numpy(xp, device):
    """The TERMA is the same on every backend, and zero on voxels without a valid depth."""
    rng = np.random.default_rng(2)
    n = 500
    rad_depths = 300 * rng.random(n)
    geo_dists = 900 + 200 * rng.random(n)
    lat = 60 * (rng.random((n, 2)) - 0.5)
    rad_depths[:3] = [np.nan, np.inf, 10.0]
    geo_dists[2] = 0.0
    fluence = np.column_stack([np.linspace(0, 200, 50), np.linspace(1.0, 0.8, 50)])
    args = (1000.0, 0.005, 2.0, 7.0, fluence, np.array([5.0, 0.0, -3.0]))
    ref = calc_terma(*args, rad_depths, geo_dists, lat, 1e-3)
    terma = calc_terma(
        *args,
        xp.asarray(rad_depths, device=device),
        xp.asarray(geo_dists, device=device),
        xp.asarray(lat, device=device),
        1e-3,
    )
    assert np.all(ref[:3] == 0) and np.any(ref > 0)
    np.testing.assert_allclose(to_numpy(terma), ref, rtol=1e-13, atol=1e-300)


def test_hardening_fit():
    """The hardening correction matches the machine depth dose better than a constant mu."""
    machine = load_from_name("photons", "Generic")
    svd = machine.get_kernel_by_energy(6.0)
    ix = int(np.argmin(np.abs(svd.kernel_ssds - machine.sad)))

    def reference(d):
        return analytical_depth_dose(
            svd.kernel_data[ix], svd.kernel_pos, svd.kernel_betas, svd.m, d
        )

    kernel = load_kernel(None, 12)
    mu0, beta = fit_hardening(kernel, reference, svd.m, 5, 100)
    assert mu0 > 0 and beta > 0
    depths = np.array([30.0, 100.0, 200.0, 300.0])
    ref = reference(depths) / reference(depths)[1]
    corrected = broad_field_depth_dose(kernel, mu0, depths, 5, beta)
    corrected = corrected / corrected[1]
    mono = broad_field_depth_dose(kernel, svd.m, depths, 5, 0)
    mono = mono / mono[1]
    assert np.max(np.abs(corrected / ref - 1)) < 0.02
    assert np.max(np.abs(corrected / ref - 1)) < np.max(np.abs(mono / ref - 1))
    np.testing.assert_allclose(
        terma_depth_factor(0.005, 0, depths), 0.005 * np.exp(-0.005 * depths), rtol=1e-14
    )


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


@pytest.fixture(scope="module")
def water_box():
    return _water_box(16, 6.0, 4)


@pytest.fixture(scope="module")
def heterogeneous_water_box():
    return _water_box(16, 6.0, 4, low_density_rows=slice(6, 9))


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


def test_forward_equals_dij_times_weights(water_box):
    """The bixel-wise forward dose equals the dose influence matrix times the weights."""
    ct, cst = water_box
    pln = _plan(6.0, 12.0, [0.0], beam_forward=False)
    stf = generate_stf(ct, cst, pln)
    dij = calc_dose_influence(ct, cst, stf, pln)
    w = np.linspace(0.5, 1.5, dij.total_num_of_bixels)
    dose = sitk.GetArrayFromImage(calc_dose_forward(ct, cst, stf, pln, w).physical_dose.values)
    ref = _dij_times_weights(dij, w, dose.shape)
    np.testing.assert_allclose(dose, ref, rtol=0, atol=1e-5 * ref.max())
    assert np.all(dose >= 0) and np.all(np.isfinite(dose))
    assert dose.max() > 0


def test_beam_forward_equals_dij_times_weights(heterogeneous_water_box):
    """Without cut-offs the beam forward dose equals the dij times the weights."""
    ct, cst = heterogeneous_water_box
    pln = _plan(6.0, 12.0, [0.0, 120.0], scatter_cutoff=1000.0, geometric_lateral_cutoff=1000.0)
    stf = generate_stf(ct, cst, pln)
    dij = calc_dose_influence(ct, cst, stf, pln)
    w = np.linspace(0.5, 1.5, dij.total_num_of_bixels)
    dose = sitk.GetArrayFromImage(calc_dose_forward(ct, cst, stf, pln, w).physical_dose.values)
    ref = _dij_times_weights(dij, w, dose.shape)
    np.testing.assert_allclose(dose, ref, rtol=0, atol=1e-5 * ref.max())
    assert np.all(dose >= 0) and np.all(np.isfinite(dose))


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
    terma = terma_depth_factor(params["mu0"], params["beta"], depth) * inv_square
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


def test_kernel_energy_mismatch_raises(tmp_path, water_box):
    """A kernel of another energy than the beam is refused."""
    data = read_mat(str(DEFAULT_KERNEL_FILE))
    data["ccKernel"]["meta"]["nominalEnergy"] = 18.0
    kernel_file = tmp_path / "kernel_18MV.mat"
    scipy.io.savemat(kernel_file, {"ccKernel": data["ccKernel"]})
    ct, cst = water_box
    pln = _plan(6.0, 12.0, [0.0], kernel_file=str(kernel_file))
    stf = generate_stf(ct, cst, pln)
    with pytest.raises(ValueError, match="does not match"):
        calc_dose_forward(ct, cst, stf, pln)
    machine = load_from_name("photons", "Generic")
    available, msg = PhotonCollapsedConeEngine.is_available(pln, machine)
    assert not available and "does not match" in msg


def _engine_dose(monkeypatch, backend_name, ct, cst, **dose_calc):
    """Forward dose of a small two-beam plan computed in the namespace of a backend."""
    gpu = backend_name in ("cupy", "torch_cuda")
    monkeypatch.setattr(settings.xp, "prefer_gpu", gpu)
    monkeypatch.setattr(
        settings.xp, "preferred_cpu_array_backend", "numpy" if gpu else backend_name
    )
    monkeypatch.setattr(
        settings.xp,
        "preferred_gpu_array_backend",
        "torch" if backend_name == "torch_cuda" else "cupy",
    )
    pln = _plan(6.0, 12.0, [0.0, 120.0], **dose_calc)
    stf = generate_stf(ct, cst, pln)
    w = np.linspace(0.5, 1.5, stf.total_number_of_bixels)
    return sitk.GetArrayFromImage(calc_dose_forward(ct, cst, stf, pln, w).physical_dose.values)


@pytest.mark.parametrize("beam_forward", [True, False])
@pytest.mark.parametrize("engine_backend", ["array_api_strict", "torch", "cupy", "torch_cuda"])
def test_engine_backend_matches_numpy(
    monkeypatch, heterogeneous_water_box, engine_backend, beam_forward
):
    """The engine gives the same dose in every namespace and on the GPU as with NumPy."""
    _skip_unavailable(engine_backend)
    dose = _engine_dose(
        monkeypatch, engine_backend, *heterogeneous_water_box, beam_forward=beam_forward
    )
    ref = _engine_dose(monkeypatch, "numpy", *heterogeneous_water_box, beam_forward=beam_forward)
    assert ref.max() > 0
    assert np.all(np.isfinite(dose)) and np.all(dose >= 0)
    np.testing.assert_allclose(dose, ref, rtol=0, atol=1e-5 * ref.max())
