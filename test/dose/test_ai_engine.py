from pathlib import Path

import SimpleITK as sitk
import numpy as np
import pytest

from pyRadPlan import xp_utils, plot_slice
from pyRadPlan.dose import calc_dose_forward, calc_dose_influence
from pyRadPlan.dose.engines import AIBeamletEngine, DoseEngineBase
from pyRadPlan.stf import validate_stf


torch = pytest.importorskip("torch", reason="torch is required for AI engine tests.")
pytest.importorskip("safetensors", reason="safetensors is required for AI engine tests.")

from pyRadPlan._settings import get_settings  # noqa: E402

DUMMY_MODEL = "dummy_dosecalc_model"


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
    result_py = sitk.GetArrayFromImage(result_py["physical_dose"])

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
