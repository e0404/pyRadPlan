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
)

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

    # this test basicly tests if fred is properly working.
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
        # TODO: 1e-3 should be possible!
        assert np.allclose(phantom_dose, dose_cube, atol=1e-2)


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
