import logging

import SimpleITK as sitk
import numpy as np
import shutil
import subprocess
import re
import pytest

from pyRadPlan.dose import calc_dose_forward, calc_dose_influence
from pyRadPlan.dose.engines import (
    ParticleFredMCEngine,
    DoseEngineBase,
    get_available_engines,
    get_engine,
    get_engine_class,
)
from pyRadPlan.dose.engines._fredmc import FRED_PARTICLES

# from pyRadPlan import generate_stf, plot_slice
from pyRadPlan.plan import IonPlan
from pyRadPlan.stf import validate_stf


def is_fred_installed():
    """Check if the FRED Monte Carlo engine is installed."""
    return shutil.which("fred") is not None


def is_latest_version():
    """Return True if installed FRED version == 3.76."""
    fred_exec = shutil.which("fred")
    if fred_exec is None:
        return False
    try:
        out = subprocess.check_output([fred_exec, "-v"], stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError:
        return False

    m = re.search(r"Version\s+(\d+\.\d+)", out)
    return bool(m and m.group(1).startswith("3.76"))


@pytest.fixture
def test_plan_fred() -> IonPlan:
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
        "engine": "FRED",  # necessary for the following to take effect
        "fred_version": "3.76.0",  # std: "3.70.0"
        "use_gpu": True,  # std: True | one has to set GPU = 1 using 'fred -config' manually!
        "calc_let": False,  # std: False
        "scorers": ["Dose"],  # std: ["Dose"]
        "room_material": "Air",  # std: Air
        "print_output": True,
        "num_histories_per_beamlet": 2e2,
        # Forward (direct) calculations use num_histories_direct, not
        # num_histories_per_beamlet; the 1e6 default makes each forward test
        # occupy the GPU for minutes. 4e4 keeps the statistical noise well below
        # the ~8e-4 systematic offset to the matRad reference (atol 1e-3) while
        # the FRED runtime stays startup-dominated (a few seconds).
        "num_histories_direct": 4e4,
        # Fail instead of blocking the GPU indefinitely (e.g. under contention).
        "execution_timeout": 120.0,
        "dose_grid": {"resolution": {"x": 10, "y": 10, "z": 10}},
        "dosimetric_lateral_cutoff": 0.995,
        "lateral_model": "single",
        # "dij_format_version": "31", # 20/30 for fred 3.70.0 and 21/31 for fred 3.76.0
        # "save_input": tmp_path,
        # "save_output": tmp_path,
        # "use_output": tmp_path,
    }
    pln.prop_opt = {"solver": "scipy"}
    return pln


def test_init_fred(test_data_protons):
    engine = ParticleFredMCEngine(test_data_protons[0])
    assert engine
    assert engine.name != None
    assert isinstance(engine, ParticleFredMCEngine)
    assert isinstance(engine, DoseEngineBase)


class Test_file_handling:
    def test_save_input_files(self, test_data_protons, test_plan_fred, tmp_path):
        pln, ct_mat, cst_mat, stf, dij_mat, _ = test_data_protons
        pln = test_plan_fred
        pln.prop_dose_calc.update(
            {
                "external_calculation": True,
                "save_input": str(tmp_path),
            }
        )

        dij = calc_dose_influence(ct_mat, cst_mat, stf, pln)

        # Check if the input files were generated
        assert (tmp_path / "inp/fred.inp").exists()
        assert (tmp_path / "inp/plan/plan.inp").exists()
        assert (tmp_path / "inp/plan/planDelivery.inp").exists()
        assert (tmp_path / "inp/regions/regions.inp").exists()
        assert (tmp_path / "inp/regions/hLut.inp").exists()
        assert (tmp_path / "inp/regions/CTpatient.mhd").exists()
        assert (tmp_path / "inp/regions/CTpatient.raw").exists()

        # check that no output was generated
        assert not (tmp_path / "out").exists(), (
            "Output files should not be generated when save_input is set."
        )

    # this test basically tests if fred is properly working.
    @pytest.mark.skipif(
        not is_fred_installed(),
        reason="FRED is not installed on this system.",
    )
    def test_save_files_generation(self, test_data_protons, test_plan_fred, tmp_path):
        pln, ct_mat, cst_mat, stf, _, _ = test_data_protons
        pln = test_plan_fred
        pln.prop_dose_calc.update(
            {
                # "external_calculation": True,
                "save_output": str(tmp_path),
            }
        )

        dij = calc_dose_influence(ct_mat, cst_mat, stf, pln)

        # Check if the output files were generated
        assert (tmp_path / "out/geom/geom0/bboxes.txt").exists()
        assert (tmp_path / "out/geom/geom0/bbox_Field_0.txt").exists()
        assert (tmp_path / "out/geom/geom0/bbox_Phantom.txt").exists()
        assert (tmp_path / "out/geom/geom0/origin.txt").exists()
        assert (tmp_path / "out/geom/geom0/reg_Field_0.txt").exists()
        assert (tmp_path / "out/geom/geom0/reg_Phantom.txt").exists()
        assert (tmp_path / "out/geom/geom0/reg_Room.txt").exists()
        assert (tmp_path / "out/geom/geom0/vertices.txt").exists()
        assert (tmp_path / "out/geom/geom1").exists()
        assert (tmp_path / "out/log").exists()
        assert (tmp_path / "out/score").exists()
        assert (tmp_path / "out/scoreij/Phantom.Dose.bin").exists()
        assert (tmp_path / "out/hu2materials.txt").exists()
        # check that no input was generated
        assert not (tmp_path / "inp").exists(), (
            "Input files should not be generated when save_input is not set."
        )


class Test_file_correctness:
    def test_input_files_generation(
        self, test_data_protons, test_data_fred_inp, test_plan_fred, tmp_path
    ):
        pln, ct_mat, cst_mat, stf, dij_mat, _ = test_data_protons
        pln = test_plan_fred
        pln.prop_dose_calc.update(
            {
                "external_calculation": True,
                "save_input": str(tmp_path),
            }
        )
        dij = calc_dose_influence(ct_mat, cst_mat, stf, pln)

        # Paths to generated files
        generated_files = {
            "inp": tmp_path / "inp/fred.inp",
            "plan": tmp_path / "inp/plan/plan.inp",
            "planDelivery": tmp_path / "inp/plan/planDelivery.inp",
            "regions": tmp_path / "inp/regions/regions.inp",
            "hLut": tmp_path / "inp/regions/hLut.inp",
            "CTpatient.mhd": tmp_path / "inp/regions/CTpatient.mhd",
            "CTpatient.raw": tmp_path / "inp/regions/CTpatient.raw",
        }

        # Compare each file's content
        for key, reference_content in test_data_fred_inp.items():
            generated_file = generated_files[key]
            assert generated_file.exists(), f"Generated file {generated_file} does not exist."

            # Handle binary files separately
            if key == "CTpatient.raw":
                with open(generated_file, "rb") as gen_file:
                    generated_content = gen_file.read()
            else:
                with open(generated_file, "r") as gen_file:
                    generated_content = gen_file.read()

            # Assert that the contents match
            assert generated_content == reference_content, f"Mismatch in file: {key}"

    @pytest.mark.skipif(
        not is_latest_version(),
        reason="FRED version 3.76 is not installed.",
    )
    def test_check_scoreij(self, test_data_protons, test_data_fred_out, test_plan_fred, tmp_path):
        from pyRadPlan.dose.engines._fredmc import read_sparse_dij_bin_v21

        _, ct_mat, cst_mat, stf_mat, _, _ = test_data_protons
        _, dij_want = test_data_fred_out
        pln = test_plan_fred
        pln.prop_dose_calc.update(
            {
                "save_output": str(tmp_path),
            }
        )
        dij = calc_dose_influence(ct_mat, cst_mat, stf_mat, pln)

        assert (tmp_path / "out/scoreij/Phantom.Dose.bin").exists()

        dij_got = read_sparse_dij_bin_v21(tmp_path / "out/scoreij/Phantom.Dose.bin").toarray()

        # Check if the content matches the expected output
        assert np.allclose(dij_got, dij_want.toarray(), atol=1e-3)

    @pytest.mark.skipif(
        not is_fred_installed(),
        reason="FRED is not installed on this system.",
    )
    def test_check_score(self, test_data_protons, test_data_fred_out, test_plan_fred, tmp_path):
        _, ct_mat, cst_mat, stf_mat, _, _ = test_data_protons
        phantom_dose, _ = test_data_fred_out
        pln = test_plan_fred

        pln.prop_dose_calc.update(
            {
                "save_output": str(tmp_path),
            }
        )
        result_py = calc_dose_forward(ct_mat, cst_mat, stf_mat, pln)

        assert (tmp_path / "out/score/Phantom.Dose.mhd").exists()
        dose_cube = sitk.GetArrayFromImage(sitk.ReadImage(tmp_path / "out/score/Phantom.Dose.mhd"))
        assert np.allclose(phantom_dose, dose_cube, atol=1e-3)


@pytest.mark.skipif(
    not is_fred_installed(),
    reason="FRED is not installed on this system.",
)
class Test_calc_dose_influence:
    def test_one_beam(self, test_data_protons, test_plan_fred):
        _, ct_mat, cst_mat, stf_mat, dij_mat, _ = test_data_protons
        pln = test_plan_fred

        # update two beam data to one beam data
        stf = stf_mat.model_dump()
        stf = {key: value[:1] for key, value in stf.items()}
        stf = validate_stf(stf)

        dij = calc_dose_influence(ct_mat, cst_mat, stf, pln)
        physical_dose_py_dense = dij.physical_dose.flat[0].toarray()  # Slice first 6 columns
        physical_dose_mat_dense = dij_mat.physical_dose.flat[0][:, :6].toarray()

        assert np.allclose(physical_dose_py_dense, physical_dose_mat_dense, atol=1e-3)

    def test_multiple_beams(self, test_data_protons, test_plan_fred):
        _, ct_mat, cst_mat, stf_mat, dij_mat, _ = test_data_protons
        pln = test_plan_fred

        dij = calc_dose_influence(ct_mat, cst_mat, stf_mat, pln)
        physical_dose_py_dense = dij.physical_dose.flat[0].toarray()
        physical_dose_mat_dense = dij_mat.physical_dose.flat[0].toarray()

        assert np.allclose(physical_dose_py_dense, physical_dose_mat_dense, atol=1e-3)

    def test_use_output(self, test_data_protons, test_plan_fred, tmp_path):
        _, ct_mat, cst_mat, stf_mat, dij_mat, _ = test_data_protons
        pln = test_plan_fred
        pln.prop_dose_calc.update(
            {
                # "save_output": str(tmp_path),
                "use_output": str("test/data/mc_fred/out"),
            }
        )

        dij = calc_dose_influence(ct_mat, cst_mat, stf_mat, pln)
        physical_dose_py_dense = dij.physical_dose.flat[0].toarray()
        physical_dose_mat_dense = dij_mat.physical_dose.flat[0].toarray()

        assert np.allclose(physical_dose_py_dense, physical_dose_mat_dense, atol=1e-2)


@pytest.mark.skipif(
    not is_fred_installed(),
    reason="FRED is not installed on this system.",
)
class Test_calc_dose_forward:
    def test_one_beam(self, test_data_protons, test_plan_fred, tmp_path):
        pln, ct_mat, cst_mat, stf_mat, _, result = test_data_protons
        pln = test_plan_fred
        # update two beam data to one beam data
        stf = stf_mat.model_dump()
        stf = {key: value[:1] for key, value in stf.items()}
        stf = validate_stf(stf)
        pln.prop_dose_calc.update(
            {
                "save_output": str(tmp_path),
            }
        )
        result_py = calc_dose_forward(ct_mat, cst_mat, stf, pln)
        result_py = sitk.GetArrayFromImage(result_py["physical_dose"])
        result_matRad_rot = np.swapaxes(result["physicalDose_beam1"], 0, 1)
        # TODO: 1e-4 should be possible!
        assert np.allclose(result_py, result_matRad_rot, atol=1e-3)
        # plot_slice(
        # ct=ct_mat,
        # cst=cst_mat,
        # overlay=result_py-result_matRad_rot,
        # view_slice=5,
        # plane="axial",
        # overlay_unit="Gy",
        # show_plot = True,
        # use_global_max = False,
        # )

    def test_multiple_beams(self, test_data_protons, test_plan_fred):
        pln, ct_mat, cst_mat, stf, dij_mat, result_mat = test_data_protons
        pln = test_plan_fred

        result_py = calc_dose_forward(ct_mat, cst_mat, stf, pln)
        result_py = sitk.GetArrayFromImage(result_py["physical_dose"])

        result_matRad_rot = np.swapaxes(result_mat["physicalDose"], 0, 1)
        # TODO: 1e-4 should be possible!
        assert np.allclose(result_py, result_matRad_rot, atol=1e-3)

    def test_use_output(self, test_data_protons, test_plan_fred, tmp_path):
        pln, ct_mat, cst_mat, stf_mat, _, result_mat = test_data_protons
        pln = test_plan_fred
        pln.prop_dose_calc.update(
            {
                # "save_output": str(tmp_path),
                "use_output": str("test/data/mc_fred/out"),
            }
        )

        result_py = calc_dose_forward(ct_mat, cst_mat, stf_mat, pln)
        result_py = sitk.GetArrayFromImage(result_py["physical_dose"])

        result_matRad_rot = np.swapaxes(result_mat["physicalDose"], 0, 1)
        # TODO: 1e-4 should be possible! (boundary is off!)
        assert np.allclose(result_py, result_matRad_rot, atol=1e-3)


def _external_fred_plan(pln, tmp_path):
    pln.prop_dose_calc = {
        "engine": "FRED",
        "external_calculation": True,
        "save_input": str(tmp_path),
        "dose_grid": {"resolution": {"x": 10, "y": 10, "z": 10}},
    }
    return pln


class TestIonRadiationModes:
    """FRED is offered under pyRadPlan's radiation mode names and maps them to its own."""

    @pytest.mark.parametrize("mode", ["protons", "helium", "carbon", "oxygen"])
    def test_fred_is_available_for_ion_modes(self, mode):
        """Every ion mode FRED simulates selects it like the factory does."""
        assert mode in ParticleFredMCEngine.possible_radiation_modes
        assert get_available_engines(mode)["FRED"] is ParticleFredMCEngine
        assert get_engine_class(mode, "FRED") is ParticleFredMCEngine

    def test_oxygen_plan_selects_fred(self):
        """An oxygen plan asking for FRED gets FRED (it used to fall back to the first)."""
        pln = IonPlan(radiation_mode="oxygen", machine="Generic")
        pln.prop_dose_calc = {"engine": "FRED"}
        assert isinstance(get_engine(pln), ParticleFredMCEngine)

    def test_oxygen_input_names_the_nuclide_and_energy_per_nucleon(
        self, test_data_oxygen, tmp_path
    ):
        """FRED reads the per-nucleon machine energy of an ion from Tu / EuFWHM."""
        pln, ct, cst, stf, _dij, _result = test_data_oxygen

        calc_dose_influence(ct, cst, stf, _external_fred_plan(pln, tmp_path))

        delivery = (tmp_path / "inp/plan/planDelivery.inp").read_text()
        assert "particle = O16\n" in delivery
        assert "Tu   = $currEnergy" in delivery
        assert "EuFWHM = $currEspread" in delivery
        assert "T    = $currEnergy" not in delivery

    def test_ion_calculation_warns_that_it_is_not_validated(
        self, test_data_oxygen, tmp_path, caplog
    ):
        pln, ct, cst, stf, _dij, _result = test_data_oxygen

        with caplog.at_level(logging.WARNING):
            calc_dose_influence(ct, cst, stf, _external_fred_plan(pln, tmp_path))

        assert "FRED's oxygen simulations are not validated" in caplog.text

    def test_proton_calculation_does_not_warn(self, test_data_protons, tmp_path, caplog):
        pln, ct, cst, stf, _dij, _result = test_data_protons

        with caplog.at_level(logging.WARNING):
            calc_dose_influence(ct, cst, stf, _external_fred_plan(pln, tmp_path))

        assert "simulations are not validated" not in caplog.text

    @pytest.mark.parametrize(("mode", "particle"), [("helium", "He4"), ("carbon", "C12")])
    def test_ion_particle_names(self, mode, particle):
        """The other ions are named by nuclide as well."""
        assert FRED_PARTICLES[mode] == particle

    @pytest.mark.parametrize("mode", ["helium", "carbon"])
    def test_machine_without_emittance_is_rejected_clearly(self, request, tmp_path, mode):
        """The generic helium and carbon machines carry no beam emittance for FRED."""
        pln, ct, cst, stf, _dij, _result = request.getfixturevalue(f"test_data_{mode}")
        with pytest.raises(ValueError, match="FRED needs the beam emittance"):
            calc_dose_influence(ct, cst, stf, _external_fred_plan(pln, tmp_path))

    def test_unsupported_radiation_mode_is_rejected(self, test_data_protons):
        """A radiation mode FRED has no particle for fails with a clear message."""
        engine = ParticleFredMCEngine(test_data_protons[0])
        engine._machine = type("Machine", (), {"radiation_mode": "electrons"})()
        with pytest.raises(ValueError, match="FRED cannot simulate radiation mode 'electrons'"):
            engine._write_plan_delivery_file()
