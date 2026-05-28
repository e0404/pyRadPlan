import numpy as np
import os
import pytest
from pathlib import Path

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
