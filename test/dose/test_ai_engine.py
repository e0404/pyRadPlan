import json
import shutil
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import pytest

from pyRadPlan import IonPlan, generate_stf, load_tg119, xp_utils, plot_slice
from pyRadPlan.ai.modelhub import _load_model
from pyRadPlan.core.xp_utils import to_namespace, to_numpy
from pyRadPlan.dose import calc_dose_forward, calc_dose_influence
from pyRadPlan.dose.engines import AIBeamletEngine, DoseEngineBase
from pyRadPlan.scenarios import CTPhaseUncertainty, RangeUncertainty, ScenarioSet, SetupUncertainty
from pyRadPlan.stf import validate_stf


torch = pytest.importorskip("torch", reason="torch is required for AI engine tests.")
pytest.importorskip("safetensors", reason="safetensors is required for AI engine tests.")

from pyRadPlan._settings import get_settings  # noqa: E402

DUMMY_MODEL = "DKFZ-RadOpt/dummy_dosecalc_model"


@pytest.fixture(autouse=True)
def _local_ai_models_dir(monkeypatch):
    """Resolve models from the committed test-fixture folder, without hitting HuggingFace.

    ``resolve_model_dir`` hands an unpinned request to ``snapshot_download``
    (see ``pyRadPlan.ai.modelhub._resolve``), only falling back to a local copy
    if that fails. Forcing offline mode here keeps these tests hermetic and
    fast, and avoids the fallback path re-downloading/re-writing version
    metadata next to the committed fixture.
    """
    models_dir = Path(__file__).resolve().parents[1] / "data" / "ai_models"
    monkeypatch.setattr(get_settings().ai, "modelhub_local_models_dir", models_dir)
    monkeypatch.setattr(get_settings().ai, "modelhub_offline", True)
    # the committed dummy model is loaded by id, like a hub model, so its code needs the opt-in
    monkeypatch.setattr(get_settings().ai, "modelhub_trust_remote_code", True)


def test_AIBeamletEngine_init(test_data_protons):
    pln = test_data_protons[0]
    pln.prop_dose_calc["engine"] = "AIBeamlet"
    pln.prop_dose_calc["model"] = DUMMY_MODEL

    engine = AIBeamletEngine(pln)
    assert engine
    assert engine.short_name == "AIBeamlet"
    assert isinstance(engine, AIBeamletEngine)
    assert isinstance(engine, DoseEngineBase)


def test_protons_cd_forward(test_data_protons):
    pln, ct, cst, stf, dij, result = test_data_protons

    pln.prop_dose_calc["engine"] = "AIBeamlet"
    pln.prop_dose_calc["model"] = DUMMY_MODEL

    if hasattr(dij, "dose_grid") and dij.dose_grid is not None:
        pln.prop_dose_calc["dose_grid"] = dij.dose_grid

    result_py = calc_dose_forward(ct, cst, stf, pln, weights=None)
    result_py = sitk.GetArrayFromImage(result_py.physical_dose.values)

    result_matRad_rot = np.swapaxes(result["physicalDose"], 0, 1)

    assert result_py.shape == result_matRad_rot.shape
    assert np.all(np.isfinite(result_py))


def test_protons_cd_influence(test_data_protons):
    """The dummy model exercises all three declared quantities (dose + physical_dose_var + LET)."""
    pln, ct, cst, stf, dij, _result = test_data_protons

    pln.prop_dose_calc["engine"] = "AIBeamlet"
    pln.prop_dose_calc["model"] = DUMMY_MODEL

    if hasattr(dij, "dose_grid") and dij.dose_grid is not None:
        pln.prop_dose_calc["dose_grid"] = dij.dose_grid

    dij_py = calc_dose_influence(ct, cst, stf, pln)

    assert dij_py.physical_dose is not None
    assert dij_py.physical_dose_var is not None
    assert dij_py.let_dose is not None

    physical_dose_dense = dij_py.physical_dose.flat[0].toarray()
    physical_dose_var_dense = dij_py.physical_dose_var.flat[0].toarray()
    let_dose_dense = dij_py.let_dose.flat[0].toarray()

    assert physical_dose_dense.shape == physical_dose_var_dense.shape == let_dose_dense.shape
    assert np.all(np.isfinite(physical_dose_dense))
    assert np.all(np.isfinite(physical_dose_var_dense))
    assert np.all(np.isfinite(let_dose_dense))
    assert np.all(physical_dose_var_dense >= 0)


def test_forward_matches_influence_matrix(test_data_protons):
    """A forward calculation must equal the influence matrix applied to the same weights."""
    pln, ct, cst, stf, dij, _result = test_data_protons
    pln.prop_dose_calc["engine"] = "AIBeamlet"
    pln.prop_dose_calc["model"] = DUMMY_MODEL

    dij_py = calc_dose_influence(ct, cst, stf, pln)
    weights = np.linspace(0.1, 1.0, dij_py.total_num_of_bixels, dtype=np.float32)

    forward = calc_dose_forward(ct, cst, stf, pln, weights=weights)
    expected = dij_py.compute_result_ct_grid(weights)

    assert np.allclose(
        sitk.GetArrayFromImage(forward.physical_dose.values),
        sitk.GetArrayFromImage(expected.physical_dose.values),
        rtol=1e-5,
        atol=1e-8,
    )


def test_construct_grid_matches_model_config(test_data_protons):
    """The beamlet-local sampling cuboid must follow the dummy model's grid_parameters.

    ``lateral_range`` is the *total* lateral width (see ``GridParametersConfig``), so a
    20 mm width with 4 mm spacing gives 5 lateral samples centered on 0 (-8..8). The
    depth axis uses a left-edge convention starting at ``depth_offset_start``.
    """
    pln, ct, cst, stf, dij, _result = test_data_protons
    pln.prop_dose_calc["engine"] = "AIBeamlet"
    pln.prop_dose_calc["model"] = DUMMY_MODEL

    engine = AIBeamletEngine(pln)
    engine._init_dose_calc(ct, cst, stf)

    grid_params = engine.model_cfg.grid_parameters
    assert grid_params.spacing == 4.0
    assert grid_params.lateral_range == 20.0
    assert grid_params.depth_offset_start == -20.0
    assert grid_params.depth_range == 40.0

    lateral = np.asarray(engine.grid_information["lateral_range"])
    depth = np.asarray(engine.grid_information["depth_range"])

    assert np.allclose(lateral, [-8.0, -4.0, 0.0, 4.0, 8.0])
    assert np.allclose(depth, np.arange(-20.0, 20.0, 4.0))

    # Full local grid is the outer product of (depth, lateral, lateral).
    assert engine.grid.shape == (depth.size * lateral.size * lateral.size, 3)

    # Bounding-box corners extend half a voxel beyond the outermost sample centers.
    corners = np.asarray(engine.corners_local)
    assert np.isclose(corners[:, 0].min(), lateral[0] - grid_params.spacing / 2)
    assert np.isclose(corners[:, 0].max(), lateral[-1] + grid_params.spacing / 2)
    assert np.isclose(corners[:, 1].min(), depth[0] - grid_params.spacing / 2)
    assert np.isclose(corners[:, 1].max(), depth[-1] + grid_params.spacing / 2)
    assert np.isclose(corners[:, 2].min(), lateral[0] - grid_params.spacing / 2)
    assert np.isclose(corners[:, 2].max(), lateral[-1] + grid_params.spacing / 2)


def test_extract_ct_cuboid_shape_matches_grid(test_data_protons):
    """The CT cuboid sampled per-ray must match the (depth, lateral, lateral) grid shape."""
    pln, ct, cst, stf, dij, _result = test_data_protons
    pln.prop_dose_calc["engine"] = "AIBeamlet"
    pln.prop_dose_calc["model"] = DUMMY_MODEL

    engine = AIBeamletEngine(pln)

    captured_shapes = []
    orig_extract = engine._extract_ct_cuboid

    def _spy_extract(ray, beam_info):
        orig_extract(ray, beam_info)
        captured_shapes.append(tuple(ray["ct_cuboid"].shape))

    engine._extract_ct_cuboid = _spy_extract

    engine.calc_dose_influence(ct, cst, stf)

    grid_params = engine.model_cfg.grid_parameters
    n_depth = int(grid_params.depth_range / grid_params.spacing)
    n_lateral = int(grid_params.lateral_range / grid_params.spacing)

    assert captured_shapes, "No rays were processed."
    assert all(shape == (n_depth, n_lateral, n_lateral) for shape in captured_shapes)


def test_quantities_match_model_config(test_data_protons):
    """Only the quantities declared in the model config end up populated in the dij."""
    pln, ct, cst, stf, dij, _result = test_data_protons
    pln.prop_dose_calc["engine"] = "AIBeamlet"
    pln.prop_dose_calc["model"] = DUMMY_MODEL

    engine = AIBeamletEngine(pln)
    dij_py = engine.calc_dose_influence(ct, cst, stf)

    assert engine.model_cfg.quantities == ["physical_dose", "physical_dose_var", "let_dose"]

    # Declared quantities are populated...
    assert dij_py.physical_dose is not None
    assert dij_py.physical_dose_var is not None
    assert dij_py.let_dose is not None

    # ...and quantities the dummy model does not declare stay unset.
    assert dij_py.alpha_dose is None
    assert dij_py.sqrt_beta_dose is None


# --- backends and devices --------------------------------------------------
#
# The engine's arrays follow settings.xp, while the model runs in its own
# framework: xp_utils.device_string bridges the two (a cupy device is not a device
# string torch understands). Each backend must therefore place the network on
# the device its arrays live on, and complete a dose calculation.

_BACKENDS = [
    pytest.param("numpy", None, "cpu", id="numpy"),
    pytest.param("array_api_strict", None, "cpu", id="array_api_strict"),
    pytest.param("numpy", "torch", "cuda", id="torch-gpu"),
    pytest.param("numpy", "cupy", "cuda", id="cupy-gpu"),
]


def _use_backend(monkeypatch, cpu_backend, gpu_backend):
    """Point settings.xp at one backend; a gpu_backend also switches the device on."""
    if gpu_backend == "torch" and not xp_utils.pytorch_gpu_available():
        pytest.skip("no CUDA-capable torch build available")
    if gpu_backend == "cupy" and not xp_utils.cupy_available():
        pytest.skip("cupy is not available")

    xp_settings = get_settings().xp
    monkeypatch.setattr(xp_settings, "prefer_gpu", gpu_backend is not None)
    monkeypatch.setattr(xp_settings, "preferred_cpu_array_backend", cpu_backend)
    monkeypatch.setattr(xp_settings, "preferred_gpu_array_backend", gpu_backend)


def _ai_pln(pln, dij):
    pln.prop_dose_calc["engine"] = "AIBeamlet"
    pln.prop_dose_calc["model"] = DUMMY_MODEL
    if getattr(dij, "dose_grid", None) is not None:
        pln.prop_dose_calc["dose_grid"] = dij.dose_grid
    return pln


@pytest.mark.parametrize("cpu_backend,gpu_backend,expected_device", _BACKENDS)
def test_network_follows_the_array_backend_device(
    test_data_protons, monkeypatch, cpu_backend, gpu_backend, expected_device
):
    """The model must be loaded onto the device the engine's arrays live on."""
    _use_backend(monkeypatch, cpu_backend, gpu_backend)
    pln, ct, cst, stf, dij, _result = test_data_protons

    engine = AIBeamletEngine(_ai_pln(pln, dij))
    engine._init_dose_calc(ct, cst, stf)

    assert next(engine.network.parameters()).device.type == expected_device


@pytest.mark.parametrize("cpu_backend,gpu_backend,expected_device", _BACKENDS)
def test_dose_influence_runs_on_every_backend(
    test_data_protons, monkeypatch, cpu_backend, gpu_backend, expected_device
):
    """A full influence calculation must complete on each backend/device combination."""
    _use_backend(monkeypatch, cpu_backend, gpu_backend)
    pln, ct, cst, stf, dij, _result = test_data_protons

    dij_py = calc_dose_influence(ct, cst, stf, _ai_pln(pln, dij))
    dose = dij_py.physical_dose.flat[0].toarray()

    assert dose.shape[0] == dij_py.dose_grid.num_voxels
    assert np.all(np.isfinite(dose))
    assert dose.max() > 0.0


def test_dose_is_the_same_on_numpy_and_torch(test_data_protons, monkeypatch):
    """Switching the array backend must not change the dose (both run the model on the CPU).

    GPU runs are excluded on purpose: the dummy model's MC-dropout ensemble draws
    from the CUDA RNG there, which does not reproduce the CPU masks.
    """
    pln, ct, cst, stf, dij, _result = test_data_protons

    _use_backend(monkeypatch, "numpy", None)
    reference = calc_dose_influence(ct, cst, stf, _ai_pln(pln, dij)).physical_dose.flat[0]

    _use_backend(monkeypatch, "torch", None)
    torch_dose = calc_dose_influence(ct, cst, stf, _ai_pln(pln, dij)).physical_dose.flat[0]

    assert np.allclose(reference.toarray(), torch_dose.toarray(), rtol=1e-4, atol=1e-8)


# --- geometry --------------------------------------------------------------
#
# The dummy model predicts a near-constant cuboid, so it cannot reveal transposed or
# mirrored axes. Here the prediction is replaced by Gaussian blobs at known positions in
# the beamlet-local frame (depth, lateral x, lateral z), and their dose centroids are
# compared with world points derived from the steering information alone.

_GEO_MODEL = "DKFZ-RadOpt/geometry_model"
_BLOB_DEPTH = 10.0  # mm beyond the isocenter plane, along the ray
_BLOB_SIGMA = 4.0
_MARKER_OFFSET = (8.0, -4.0)  # mm along the cuboid's lateral axes 1 and 2
_MARKER_SIGMA = 3.0


@pytest.fixture
def geometry_model(tmp_path, monkeypatch):
    """Provide the dummy model with a sampling cuboid large enough for the blobs."""
    model_dir = tmp_path / "models" / _GEO_MODEL
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "data" / "ai_models" / DUMMY_MODEL, model_dir
    )
    config_file = model_dir / "model_config.json"
    config = json.loads(config_file.read_text())
    config["metadata"]["dose_calc"]["grid_parameters"] = {
        "spacing": 2,
        "lateral_range": 40,
        "depth_offset_start": -40,
        "depth_range": 80,
    }
    config["metadata"]["dose_calc"]["quantities"] = ["physical_dose"]
    config_file.write_text(json.dumps(config))
    monkeypatch.setattr(get_settings().ai, "modelhub_local_models_dir", tmp_path / "models")
    return _GEO_MODEL


def _blob_predictor(with_marker):
    """Replace ``_predict`` by a blob on the beamlet axis, optionally plus a marker.

    The amplitude follows the bixel energy, so a mix-up of bixels within a ray shows.
    """

    def predict(self, ct_cuboid, energies):
        depth = to_numpy(self.grid_information["depth_range"]).astype(np.float64)
        lateral = to_numpy(self.grid_information["lateral_range"]).astype(np.float64)
        d, x, z = np.meshgrid(depth, lateral, lateral, indexing="ij")
        cuboid = np.exp(-((d - _BLOB_DEPTH) ** 2 + x**2 + z**2) / (2 * _BLOB_SIGMA**2))
        if with_marker:
            dx, dz = _MARKER_OFFSET
            r2 = (d - _BLOB_DEPTH) ** 2 + (x - dx) ** 2 + (z - dz) ** 2
            cuboid = cuboid + 0.5 * np.exp(-r2 / (2 * _MARKER_SIGMA**2))
        scale = to_numpy(energies).astype(np.float64) / 100.0
        batch = (scale[:, None, None, None] * cuboid[None]).astype(np.float32)
        return {"physical_dose": to_namespace(self.xp, batch, device=self.device)}

    return predict


def _geometry_case(model, max_batch_size=32):
    ct, cst = load_tg119()
    pln = IonPlan(radiation_mode="protons", machine="Generic")
    pln.prop_stf = {"gantry_angles": [40.0], "couch_angles": [25.0], "bixel_width": 20.0}
    pln.prop_dose_calc = {
        "engine": "AIBeamlet",
        "model": model,
        "dose_grid": {"resolution": [3.0, 4.0, 5.0]},
        "max_batch_size": max_batch_size,
    }
    return pln, ct, cst, generate_stf(ct, cst, pln)


def _dose_centroid(image):
    """Dose-weighted centroid in world coordinates, from the image's own geometry."""
    dose = sitk.GetArrayFromImage(image).astype(np.float64)
    assert dose.sum() > 0, "no dose deposited"
    k, j, i = np.nonzero(dose)
    weights = dose[k, j, i]
    index = weights @ np.stack((i, j, k), axis=1) / weights.sum()
    return np.asarray(image.TransformContinuousIndexToPhysicalPoint(index.tolist()))


def _bev_to_world(beam):
    """Linear map from BEV to world (relative to the isocenter), fitted to the stf."""
    bev = [np.ravel(beam.source_point_bev)] + [np.ravel(r.ray_pos_bev) for r in beam.rays]
    world = [np.ravel(beam.source_point)] + [np.ravel(r.ray_pos) for r in beam.rays]
    transposed, *_ = np.linalg.lstsq(np.asarray(bev, float), np.asarray(world, float), rcond=None)
    return transposed.T


def _unit(v):
    return v / np.linalg.norm(v)


_CPU_BACKENDS = [
    pytest.param("numpy", id="numpy"),
    pytest.param("array_api_strict", id="array_api_strict"),
    pytest.param("torch", id="torch"),
]


@pytest.mark.parametrize("cpu_backend", _CPU_BACKENDS)
def test_dose_lands_where_the_beam_geometry_puts_it(geometry_model, monkeypatch, cpu_backend):
    """Blob and marker centroids match the oblique beam geometry on an anisotropic grid.

    The blob checks depth and the ray direction; the marker, offset unequally along both
    lateral axes, checks that cuboid axis 1 follows BEV x and axis 2 BEV z (no swap or
    mirror). The marker is isolated as the difference of two calculations, since the
    engine is linear in the predicted cuboid.
    """
    _use_backend(monkeypatch, cpu_backend, None)
    pln, ct, cst, stf = _geometry_case(geometry_model)
    beam = stf.beams[0]

    monkeypatch.setattr(AIBeamletEngine, "_predict", _blob_predictor(with_marker=False))
    blob_dij = calc_dose_influence(ct, cst, stf, pln)
    monkeypatch.setattr(AIBeamletEngine, "_predict", _blob_predictor(with_marker=True))
    both_dij = calc_dose_influence(ct, cst, stf, pln)

    iso = np.ravel(beam.iso_center).astype(float)
    source = np.ravel(beam.source_point).astype(float)
    bev_to_world = _bev_to_world(beam)

    lateral = [np.linalg.norm(np.ravel(r.ray_pos_bev)) for r in beam.rays]
    assert min(lateral) < 5.0 and max(lateral) > 30.0
    first_bixel = np.cumsum([0] + [len(r.beamlets) for r in beam.rays])

    for j in range(len(beam.rays)):
        direction = _unit(np.ravel(beam.rays[j].ray_pos).astype(float) - source)
        expected_blob = iso + source + (beam.sad + _BLOB_DEPTH) * direction
        # BEV lateral axes as seen from the source, perpendicular to this ray
        axis_x, axis_z = (
            _unit(a - (a @ direction) * direction)
            for a in (bev_to_world[:, 0], bev_to_world[:, 2])
        )
        expected_marker = expected_blob + _MARKER_OFFSET[0] * axis_x + _MARKER_OFFSET[1] * axis_z

        weights = np.zeros(blob_dij.total_num_of_bixels)
        weights[first_bixel[j]] = 1.0
        blob = blob_dij.compute_result_dose_grid(weights).physical_dose.values
        both = both_dij.compute_result_dose_grid(weights).physical_dose.values
        marker = sitk.Subtract(both, blob)

        assert np.linalg.norm(_dose_centroid(blob) - expected_blob) < 0.5, f"ray {j}"
        assert np.linalg.norm(_dose_centroid(marker) - expected_marker) < 0.5, f"ray {j}"


def test_beamlet_batches_do_not_change_the_dose(geometry_model, monkeypatch):
    """Predicting a ray in batches of four beamlets gives the same matrix as all at once."""
    _use_backend(monkeypatch, "numpy", None)
    monkeypatch.setattr(AIBeamletEngine, "_predict", _blob_predictor(with_marker=True))

    pln, ct, cst, stf = _geometry_case(geometry_model, max_batch_size=None)
    assert max(len(r.beamlets) for r in stf.beams[0].rays) > 4
    whole = calc_dose_influence(ct, cst, stf, pln).physical_dose.flat[0]

    pln.prop_dose_calc["max_batch_size"] = 4
    batched = calc_dose_influence(ct, cst, stf, pln).physical_dose.flat[0]

    assert whole.nnz > 0
    assert np.allclose(whole.toarray(), batched.toarray(), rtol=1e-6, atol=0.0)


# --- scenarios -------------------------------------------------------------


def _scenario_set(*, setup=None, range_=None):
    dims = (
        CTPhaseUncertainty(phases=[0], probabilities=[1.0]),
        SetupUncertainty(),
        RangeUncertainty(),
    )
    points = {}
    if setup is not None:
        points["setup"] = np.asarray(setup, dtype=float)
    if range_ is not None:
        points["range"] = np.asarray(range_, dtype=float)
    return ScenarioSet.from_points(dims, points)


def test_setup_scenarios_are_computed(test_data_protons):
    """Each setup shift moves the beamlets, so its matrix differs from the nominal one."""
    pln, ct, cst, stf, dij, _result = test_data_protons
    engine = AIBeamletEngine(_ai_pln(pln, dij))

    dij_py = engine.calc_dose_influence(
        ct, cst, stf, scenarios=_scenario_set(setup=[(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)])
    )
    nominal, shifted = dij_py.physical_dose.flat[0], dij_py.physical_dose.flat[1]

    assert nominal.shape == shifted.shape
    assert nominal.nnz > 0 and shifted.nnz > 0
    assert (nominal != shifted).nnz > 0


def test_range_scenarios_are_rejected(test_data_protons):
    """The model predicts from the CT alone, so a range error cannot be applied."""
    pln, ct, cst, stf, dij, _result = test_data_protons
    engine = AIBeamletEngine(_ai_pln(pln, dij))

    with pytest.raises(ValueError, match=r"\['range'\] which are not supported"):
        engine.calc_dose_influence(
            ct, cst, stf, scenarios=_scenario_set(range_=[(0.0, 0.0), (3.5, 0.0)])
        )


# --- remote code -----------------------------------------------------------


def test_hub_model_code_needs_the_trust_opt_in(test_data_protons, monkeypatch):
    """Without the opt-in the engine refuses a model by id before anything is downloaded."""
    pln, ct, cst, stf, dij, _result = test_data_protons
    monkeypatch.setattr(get_settings().ai, "modelhub_trust_remote_code", False)
    monkeypatch.setattr(get_settings().ai, "modelhub_offline", False)

    def _no_download(**kwargs):
        raise AssertionError("untrusted model code must not even be downloaded")

    monkeypatch.setattr(_load_model, "resolve_model_dir", _no_download)
    pln = _ai_pln(pln, dij)
    pln.prop_dose_calc["model"] = "SomeOrg/hub-only-model"

    with pytest.raises(ValueError, match="modelhub_trust_remote_code"):
        AIBeamletEngine(pln).calc_dose_influence(ct, cst, stf)
