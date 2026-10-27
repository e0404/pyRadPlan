import numpy as np
import os
import pytest
from pathlib import Path

import logging

from pyRadPlan.biology.models import Wedenberg, create_bio_model
from pyRadPlan.dose import calc_dose_forward
from pyRadPlan.dose.engines import (
    ParticleTOPASMCEngine,
    DoseEngineBase,
)

from pyRadPlan.plan import IonPlan
from datetime import datetime
import shutil


@pytest.fixture
def test_plan_topas() -> IonPlan:
    pln = IonPlan(radiation_mode="protons", machine="Generic")
    pln.prop_stf = {
        "gantry_angles": [0, 180],  # define gantry angles for n beams
        "couch_angles": [0, 0],
        "longitudinal_spot_spacing": 2.0,
        "iso_center": np.array([[-5.0, -5.0, -5.0], [-5.0, -5.0, -5.0]]),  # two beams
        "num_of_beams": 2,
        "bixel_width": 10,
        "add_margin": 1,
    }
    pln.prop_dose_calc = {
        "engine": "TOPAS",  # necessary for the following to take effect
        "external_calculation": True,
    }
    pln.prop_opt = {"solver": "scipy"}
    return pln


def test_init_topas(test_data_protons):
    engine = ParticleTOPASMCEngine(test_data_protons[0])
    assert engine
    assert engine.name != None
    assert isinstance(engine, ParticleTOPASMCEngine)
    assert isinstance(engine, DoseEngineBase)


class Test_file_handling:
    def test_save_input_files(self, test_data_protons, test_plan_topas, tmp_path):
        _, ct_mat, cst_mat, stf, _, _ = test_data_protons
        pln = test_plan_topas
        pln.prop_dose_calc.update({"simu_dir": tmp_path / datetime.now().strftime("%Y-%m-%d")})
        result = calc_dose_forward(ct_mat, cst_mat, stf, pln)

        # Check if the input files were generated
        assert (tmp_path / datetime.now().strftime("%Y-%m-%d") / "pyRadPlan_cube.dat").exists()
        assert (tmp_path / datetime.now().strftime("%Y-%m-%d") / "pyRadPlan_cube.txt").exists()
        for beams in range(pln.prop_stf["num_of_beams"]):
            for run in range(pln.prop_stf.get("num_of_runs", 1)):
                assert (
                    tmp_path
                    / datetime.now().strftime("%Y-%m-%d")
                    / f"beamSetup_pyRadPlan_plan_field{beams}.txt"
                ).exists()
                assert (
                    tmp_path
                    / datetime.now().strftime("%Y-%m-%d")
                    / f"pyRadPlan_plan_field{beams}_run{run}.txt"
                ).exists()

        shutil.rmtree(tmp_path)


def test_default_simu_dir_is_in_user_data_dir(test_data_protons, tmp_path, monkeypatch):
    """The engine must never write next to the installed package."""
    from pyRadPlan.core import get_data_dir

    monkeypatch.setenv("PYRADPLAN_DATA_DIR", str(tmp_path))
    engine = ParticleTOPASMCEngine(test_data_protons[0])
    simu_dir = Path(engine.simu_dir)
    assert simu_dir.parent == get_data_dir() / "topas"
    assert simu_dir.name == datetime.now().strftime("%Y-%m-%d")


# ------------------------------------------------------------- LET for biological models
def _topas_plan(pln, bio_model, **settings):
    pln.bio_model = bio_model
    pln.prop_dose_calc = {"engine": "TOPAS", "num_runs": 2, **settings}
    return pln


def _run_files_text(simu_dir):
    return "".join(p.read_text() for p in Path(simu_dir).glob("pyRadPlan_plan_field*_run*.txt"))


def test_topas_scores_let_for_a_let_based_model(test_data_protons, tmp_path):
    """A LET-based model switches on the LET scorer and allocates its influence matrices."""
    pln, ct, cst, stf, _dij, _result = test_data_protons
    engine = ParticleTOPASMCEngine(_topas_plan(pln, "WED", simu_dir=tmp_path))

    dij = engine._init_dose_calc(ct, cst, stf)

    assert engine.provided_quantities() == ["physical_dose", "let"]
    # calc_let="auto" stores the LET a model needs, as with FRED
    assert engine._calc_bio_dose and engine._calc_let and engine._use_let_kernel
    assert engine._bio_influence_names == ("alpha_dose", "sqrt_beta_dose")
    assert dij["bio_model"] == Wedenberg()
    assert engine.scorer["LET"]
    assert not ParticleTOPASMCEngine.scorer["LET"], "the class default must stay untouched"
    for name in ("alpha_dose", "sqrt_beta_dose"):
        assert dij[name].flat[0].shape == dij["physical_dose"].flat[0].shape
        assert f"{name}_var" not in dij


def test_topas_input_has_the_let_scorer_only_when_needed(test_data_protons, tmp_path):
    """The run files score LET for a LET-based model, not for a constant RBE."""
    pln, ct, cst, stf, _dij, _result = test_data_protons
    with_let, without_let = tmp_path / "with_let", tmp_path / "without_let"

    calc_dose_forward(ct, cst, stf, _topas_plan(pln, "MCN", simu_dir=with_let))
    calc_dose_forward(ct, cst, stf, _topas_plan(pln, "constant_rbe", simu_dir=without_let))

    assert '= "ProtonLET"' in _run_files_text(with_let)
    assert "ProtonLET" not in _run_files_text(without_let)


@pytest.mark.parametrize(
    ("calc_bio_dose", "calc_let", "bio", "let"),
    [
        ("auto", "auto", True, True),
        ("auto", False, True, False),
        (False, "auto", False, True),
        (False, False, False, False),
    ],
)
def test_topas_quantity_switches_follow_fred(
    test_data_protons, tmp_path, calc_bio_dose, calc_let, bio, let
):
    """calc_bio_dose / calc_let resolve as for FRED: "auto" LET when the model needs it."""
    pln, ct, cst, stf, _dij, _result = test_data_protons
    pln = _topas_plan(
        pln, "WED", simu_dir=tmp_path, calc_bio_dose=calc_bio_dose, calc_let=calc_let
    )
    engine = ParticleTOPASMCEngine(pln)

    dij = engine._init_dose_calc(ct, cst, stf)

    assert (engine._calc_bio_dose, engine._calc_let) == (bio, let)
    assert engine.scorer["LET"] is (bio or let)
    assert ("alpha_dose" in dij) is bio
    assert ("let_dose" in dij) is let


def test_topas_drops_a_let_model_for_ions_it_scores_no_let_for(test_data_carbon, caplog):
    """The ProtonLET scorer ignores ions, so a carbon LET model runs as physical dose."""
    _pln, _ct, cst, _stf, _dij, _result = test_data_carbon
    engine = ParticleTOPASMCEngine()
    engine._machine = DoseEngineBase.load_machine("carbon", "Generic")
    engine.bio_model = "LSM"
    (issue,) = create_bio_model("LSM").check_tissues(engine._machine, cst, engine=engine)
    assert "nor scores LET for carbon (only for protons)" in issue.message

    dij = {}
    with caplog.at_level(logging.WARNING):
        engine._resolve_bio_model(dij, "carbon", cst)

    assert dij["bio_model"] is None
    assert issue.message in caplog.text


def _write_topas_output(folder, quantity, cubes, num_runs):
    for beam_ix, cube in enumerate(cubes):
        for run_ix in range(num_runs):
            label = f"score_pyRadPlan_plan_field{beam_ix}_run{run_ix}_{quantity}.bin"
            np.asarray(cube, dtype=np.float64).tofile(folder / label)


def test_topas_derives_biological_matrices_from_the_scored_let(test_data_protons, tmp_path):
    """let_dose = Σ_runs LETd·dose, and alpha/beta follow the model at the beam's LETd."""
    pln, ct, cst, stf, _dij, _result = test_data_protons
    settings = {"calc_let": True, "simu_dir": tmp_path / "inp"}
    probe = ParticleTOPASMCEngine(_topas_plan(pln, "WED", **settings))
    probe._init_dose_calc(ct, cst, stf)
    shape = tuple(probe.dose_grid.dimensions[::-1])
    num_beams = len(stf.beams)

    rng = np.random.default_rng(3)
    doses = [rng.random(shape) for _ in range(num_beams)]
    lets = [1.0 + 10.0 * rng.random(shape) for _ in range(num_beams)]
    _write_topas_output(tmp_path, "physicalDose", doses, num_runs=2)
    _write_topas_output(tmp_path, "LET", lets, num_runs=2)

    # TOPAS computes one column per beam, i.e. the forward calculation's dij
    engine = ParticleTOPASMCEngine(
        _topas_plan(pln, "WED", external_calculation=str(tmp_path), **settings)
    )
    engine._calc_dose_direct = True
    dij = engine._calc_dose(ct, cst, stf)

    dose = np.asarray(dij.physical_dose.flat[0])
    let_dose = np.asarray(dij.let_dose.flat[0])
    inside = dose > 0
    assert inside.any()
    expected_let = np.stack([let.ravel() for let in lets], axis=1)
    assert np.allclose(let_dose[inside] / dose[inside], expected_let[inside], rtol=1e-5)
    assert np.all(let_dose[~inside] == 0)

    alpha_x = np.asarray(dij.alphax)[:, 0]
    beta_x = np.asarray(dij.betax)[:, 0]
    for beam_ix in range(num_beams):
        voxels = inside[:, beam_ix]
        alpha, beta = Wedenberg().alpha_beta(
            alpha_x[voxels], beta_x[voxels], {"let": expected_let[voxels, beam_ix]}
        )
        d = dose[voxels, beam_ix]
        assert np.allclose(np.asarray(dij.alpha_dose.flat[0])[voxels, beam_ix], d * alpha)
        assert np.allclose(
            np.asarray(dij.sqrt_beta_dose.flat[0])[voxels, beam_ix], d * np.sqrt(beta)
        )

    result = dij.compute_result_dose_grid(np.ones(dij.total_num_of_bixels))
    assert np.all(np.isfinite(np.asarray(result["rbe_x_dose"])))
    assert np.asarray(result["let"]).max() > 1.0


def test_topas_reports_missing_let_output(test_data_protons, tmp_path):
    """Output simulated without the LET scorer cannot feed a LET-based model."""
    pln, ct, cst, stf, _dij, _result = test_data_protons
    probe = ParticleTOPASMCEngine(_topas_plan(pln, "WED", simu_dir=tmp_path / "inp"))
    probe._init_dose_calc(ct, cst, stf)
    shape = tuple(probe.dose_grid.dimensions[::-1])
    _write_topas_output(tmp_path, "physicalDose", [np.ones(shape)] * len(stf.beams), 2)

    pln = _topas_plan(pln, "WED", external_calculation=str(tmp_path))
    with pytest.raises(ValueError, match="TOPAS LET output is missing"):
        calc_dose_forward(ct, cst, stf, pln)
